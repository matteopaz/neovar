#!/bin/bash -l

#SBATCH --job-name=neowise-clustering-mpaz
#SBATCH --array=0-11%12

srun python makemaghistogram.py