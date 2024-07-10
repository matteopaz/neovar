#!/bin/bash -l

#SBATCH --job-name=neowise-clustering-mpaz
#SBATCH --exclusive
#SBATCH --array=0-9%10

srun python instance.py