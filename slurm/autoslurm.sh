#!/bin/bash
#SBATCH --job-name="dreamerv3cem_3"
#SBATCH --error=
#SBATCH --output=
#SBATCH --partition="ei-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="short"

cd /srv/...
source ~/.bashrc

srun dreamerv3/main.py --configs crafter size50m --logdir ./logs --jax.platform gpu