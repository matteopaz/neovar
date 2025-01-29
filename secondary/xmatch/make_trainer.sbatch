#!/bin/bash -l

#SBATCH --job-name=xmatch-gaia
#SBATCH --time=0-1:00:00
#SBATCH --array=0-7%8

srun python make_trainer.py