#!/bin/bash
#SBATCH --job-name="dreamerv3cem_3"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="long"

echo "Starting job $SLURM_JOB_ID using directory $LOGDIR and name $NAME"
echo "Available devices: $CUDA_VISIBLE_DEVICES; $LD_LIBRARY_PATH; $CUDA_HOME, $PATH"	

cd ~/flash/dreamerv3-cem
source ~/miniconda3/bin/activate
conda activate dreamerv3-cem

echo $(conda info --env)

srun python dreamerv3/main.py --configs crafter size50m --logdir $LOGDIR --logid $NAME --jax.platform gpu