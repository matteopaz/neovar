#!/bin/bash -l

#SBATCH --job-name=postprocess
#SBATCH --time=0-8:00:00
#SBATCH --exclusive
#SBATCH --array=0-9%10

srun python postprocess.py