#!/bin/bash -l

#SBATCH --job-name=neowise-clustering-mpaz
#SBATCH --ntasks=4
#SBATCH --exclusive

python3 init_clustering.py

for partition_id in {0..3}
do
    srun --ntasks=1 sleep 1 & python3 print_test.py --partition_id $partition_id &
done