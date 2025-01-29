#!/bin/bash -l

#SBATCH --job-name=cache
#SBATCH --time=0-1:00:00
#SBATCH --cpus-per-task=16
#SBATCH --array=0-9%10

srun python cache.py