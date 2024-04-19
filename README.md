## DDP Starter

This is a collection of tests that were done during the MCH hackathon in April 2024. The work is based on a demo found here: https://github.com/sadamov/ddp_starter

## Contents
We mostly worked with [lightning](https://lightning.ai/docs/pytorch/stable/) and tested its scaling capabilites to multiple GPUs. We also tested [mlflow](https://mlflow.org/) for logging certain metrics that might be of interest (e.g. GPU usage). 

### mlflow-lightning
This is a extension of a demo we found [here](https://docs.ray.io/en/latest/train/getting-started-pytorch-lightning.html)
### pytorch-lightning-ray
todo
### slurm-ray-cluster
todo

## Pre-requisites

- [Mamba](https://mamba.readthedocs.io/en/latest/installation.html)
- [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
- Access to a Slurm cluster (we did everything on balfrin.cscs.ch)

### Installation (todo)
Copy latest code from github:
```
git clone git@github.com:LeonardKnirsch/ddp_starter.git
cd ddp_starter
```
Create a new conda environment and install dependencies:
```
mamba env create -f environment.yml
```

### Usage
```
cd <folder of interest>
# Check the .sh script to change the number of nodes/GPUs you need. (Don't forget that you are running in a shared cluster and you could block users/important jobs.)
sbatch <.sh script>
```
TODO
Then check out the logs in `./lightning_logs` to see if the run was successful. The
`metrics.csv` contains the training and validation losses across all epochs.
```
`Trainer.fit` stopped: `max_epochs=10` reached.
```
means that the run was successful.
TODO