#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --time=00:10:00
#SBATCH --partition=normal
#SBATCH --output=lightning_logs/%j_test_ray_cluster.out
#SBATCH --error=lightning_logs/%j_test_ray_cluster.err
### This script works for any number of nodes, Ray will find and manage all resources
#SBATCH --nodes=3
### Give all resources on each node to a single Ray task, ray can manage the resources internally
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4


# Load modules or your own conda environment here
conda activate ray-slurm

head_node=$(hostname)
head_node_ip=$(hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
fi
port=6379

echo "STARTING HEAD at $head_node"
echo "Head node IP: $head_node_ip"
srun --nodes=1 --ntasks=1 -w $head_node start-head.sh $head_node_ip &
sleep 10

worker_num=$(($SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node
echo "STARTING $worker_num WORKERS" 
srun -n $worker_num --nodes=$worker_num --ntasks-per-node=1 --exclude $head_node start-worker.sh $head_node_ip:$port &
sleep 5

#### call your code below
conda run -n rayenv python examples/test_pytorch-lightning_ray.py

exit
