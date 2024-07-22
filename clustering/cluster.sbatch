#!/bin/bash -l

#SBATCH --job-name=neowise-clustering-mpaz
#SBATCH --exclusive
#SBATCH --array=0-30%11
#SBATCH --time=24:00:00

srun python instance_special.py