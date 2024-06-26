import os
import time

import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor, Normalize, Compose
import lightning.pytorch as pl

import ray.train.lightning
from ray.train.torch import TorchTrainer

# init ML flow. Autolog makes your life easy
# you only want to log on the 0th gpu on the 0th node as to have only one experiment
# tracked in mlflow
# look at the results in a browser window via typing in the terminal:
# mlflow server --host 127.0.0.1 --port 8044 # any port in the 8xxx should do
# and then allow VSCode to open the browser window
import mlflow
nodeid = int(os.environ.get('SLURM_NODEID'))
procid = int(os.environ.get('SLURM_PROCID'))
print(f'nodeid {nodeid}, procid {procid}')
if nodeid == 0 and procid == 0:
    print('logging enabled')
    os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
    mlflow.pytorch.autolog()

# Model, Loss, Optimizer
class ImageClassifier(pl.LightningModule):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.model = resnet18(num_classes=10)
        self.model.conv1 = torch.nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.forward(x)
        loss = self.criterion(outputs, y)
        self.log("loss", loss, on_step=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001)

# get number of devices and nodes
device_count = torch.cuda.device_count()
num_nodes = int(os.environ.get('SLURM_JOB_NUM_NODES'))


def train_func():
    # Data
    transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    train_data = FashionMNIST(root='./data', train=True, download=True, transform=transform)
    train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True)

    # Training
    model = ImageClassifier()
    # [1] Configure PyTorch Lightning Trainer.
    trainer = pl.Trainer(
        max_epochs=10,
        devices=device_count,
        accelerator="cuda",
        strategy=ray.train.lightning.RayDDPStrategy(),
        plugins=[ray.train.lightning.RayLightningEnvironment()],
        callbacks=[ray.train.lightning.RayTrainReportCallback()],
        # [1a] Optionally, disable the default checkpointing behavior
        # in favor of the `RayTrainReportCallback` above.
        enable_checkpointing=False,
        #strategy="ddp",
        log_every_n_steps=1,
        num_nodes=num_nodes,
    )
    trainer = ray.train.lightning.prepare_trainer(trainer)
    trainer.fit(model, train_dataloaders=train_dataloader)

# [2] Configure scaling and resource requirements.
scaling_config = ray.train.ScalingConfig(num_workers=device_count, use_gpu=True)

# [3] Launch distributed training job.
trainer = TorchTrainer(
    train_func,
    scaling_config=scaling_config,
    # [3a] If running in a multi-node cluster, this is where you
    # should configure the run's persistent storage that is accessible
    # across all worker nodes.
    # run_config=ray.train.RunConfig(storage_path="s3://..."),
)
start = time.time()
result: ray.train.Result = trainer.fit()
end = time.time()
total = end - start
print(f"{num_nodes*device_count} GPUs took {total} seconds.")

# [4] Load the trained model.
#with result.checkpoint.as_directory() as checkpoint_dir:
#    model = ImageClassifier.load_from_checkpoint(
#        os.path.join(
#            checkpoint_dir,
#            ray.train.lightning.RayTrainReportCallback.CHECKPOINT_NAME,
#        ),
#    )