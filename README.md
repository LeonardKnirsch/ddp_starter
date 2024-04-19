## DDP Starter

This is a collection of tests that were done during the MCH hackathon in April 2024. The work is based on a demo found here: https://github.com/sadamov/ddp_starter

## Contents
We mostly worked with [lightning](https://lightning.ai/docs/pytorch/stable/) and tested its scaling capabilites to multiple GPUs. We also tested [mlflow](https://mlflow.org/) for logging certain metrics that might be of interest (e.g. GPU usage). 

### lightning
This is a extension of a demo we found [here](https://docs.ray.io/en/latest/train/getting-started-pytorch-lightning.html).
### lightning-ray
This is a extension of the same demo as above but this time using Ray.
### lightning-ray-multinode
This combines the above script from lightning-ray with a multi-node slurm setup, as demonstrated [here](https://github.com/NERSC/slurm-ray-cluster).

## Pre-requisites

- [Mamba](https://mamba.readthedocs.io/en/latest/installation.html)
- [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
- Access to a Slurm cluster (we did everything on balfrin.cscs.ch)

### Installation
Copy latest code from github:
```
git clone git@github.com:LeonardKnirsch/ddp_starter.git
cd ddp_starter
```

Create a new conda environment and install dependencies:
```
mamba env create -f pinned_env.yml # we have experienced some unresolved dependency issues with environment.yml for now
```

Change directory to the demo folder you want to install an environment for:
```
# cd <lightning/lightning-ray/lightning-ray-multinode>
cd lightning
```


### Usage
```
cd <folder of interest>
# Check the .sh script to change the number of nodes/GPUs you need. (Don't forget that you are running in a shared cluster and you could block users/important jobs.)
# If you are in lightning-ray-multinode, then use test-ray-cluster.sh as the sbatch script.
sbatch <.sh script>
```

Then check out the logs in `./lightning_logs` to see if the run was successful. 
```
`Trainer.fit` stopped: `max_epochs=10` reached.
```
means that the run was successful. Alternatively, you can use Mlflow metrics, which are set up for the lightning/ and lightning-ray/ examples.
