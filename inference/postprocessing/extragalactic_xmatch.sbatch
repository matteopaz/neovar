#!/bin/bash -l

#SBATCH --job-name=egalxmatch
#SBATCH --time=0-1:00:00
#SBATCH --exclusive
#SBATCH --array=0-7%8

srun python extragalactic_xmatch.py