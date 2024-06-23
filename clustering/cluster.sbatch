#!/bin/bash -l

#SBATCH --job-name=neowise-clustering-mpaz
#SBATCH --exclusive
#SBATCH --array=0-123%9

srun python instance.py