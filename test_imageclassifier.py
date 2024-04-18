import os

import torch
from torchvision.models import resnet18
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, TensorDataset
import lightning.pytorch as pl
import time

class IrisDataModule(pl.LightningDataModule):
    """Data module for the Iris dataset."""

    def __init__(self):
        """Initializes the data module."""
        super().__init__()
        iris = load_iris()
        X_train, X_val, y_train, y_val = train_test_split(
            iris.data, iris.target, test_size=0.2, random_state=42)
        self.train_data = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train))
        self.val_data = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val))

    def train_dataloader(self):
        """Returns a DataLoader for the training data."""
        return DataLoader(self.train_data, batch_size=32, num_workers=16)

    def val_dataloader(self):
        """Returns a DataLoader for the validation data."""
        return DataLoader(self.val_data, batch_size=32, num_workers=16)

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

if __name__ == "__main__":
    """Main function to set up and train the model."""

    #torch.set_float32_matmul_precision('high')
    device_count = torch.cuda.device_count()
    num_nodes = int(os.environ.get('SLURM_JOB_NUM_NODES'))

    print("num_devices: ", device_count)
    print("num_nodes: ",num_nodes)

    # Data
    transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    train_data = FashionMNIST(root='/scratch/mch/bverena/FashionMNIST', train=True, download=True, transform=transform)
    train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True)

    # Training
    model = ImageClassifier()

    trainer = pl.Trainer(
        max_epochs=10,
        devices=device_count,
        accelerator='cuda',
        strategy="ddp",
        log_every_n_steps=1,
        num_nodes=num_nodes,
    )
    #trainer.fit(model, data)
    start = time.time()
    trainer.fit(model, train_dataloaders=train_dataloader)
    end = time.time()
    total = end - start
    print(f"{num_nodes*device_count} GPUs took {total} seconds.")
