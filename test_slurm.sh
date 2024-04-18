#!/bin/bash -l
#SBATCH --job-name=pl_ddp_test
#SBATCH --output=lightning_logs/pl_ddp_test.out
#SBATCH --error=lightning_logs/pl_ddp_test.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:10:00
#SBATCH --partition=normal

# Load necessary modules
conda activate ddp_starter

# Run the script
#srun -ul python test_ddp.py
srun -ul python test_imageclassifier.py
