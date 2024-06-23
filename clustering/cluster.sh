#!/bin/bash -l

#SBATCH --job-name=neowise-clustering-mpaz
#SBATCH --exclusive
#SBATCH --array=0-123%8

python init.py

srun python dummyinstance.py