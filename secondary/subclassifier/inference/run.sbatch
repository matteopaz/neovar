#!/bin/bash -l
#SBATCH --job-name="varclass"
#SBATCH --time=12:00:00
#SBATCH --array=0-9%10
#SBATCH --exclusive

srun python inst.py