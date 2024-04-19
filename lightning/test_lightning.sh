#!/bin/bash -l

# --ntasks-per-node and --gres=gpu must be the same number
# you get a total of nodes*ntasks GPUs

#SBATCH --job-name=pl_ddp_test
#SBATCH --output=lightning_logs/%jpl_ddp_test.out
#SBATCH --error=lightning_logs/%jpl_ddp_test.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --partition=normal

# Load necessary modules
conda activate friss_oder_stirb

# Run the script
#srun -ul python test_ddp.py
srun -ul python lightning/test_mlflow_lightning.py
