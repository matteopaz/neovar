#!/bin/bash -l

#SBATCH --job-name=neowise-clustering-mpaz
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time="01:00:00"

read -p "Reload and dump schema? [y/n]" yn

if [ $yn="y" ]; then
srun --ntasks=1 python3 init_clustering.py &
wait
fi

for partition_id in {0..0}
do
    srun --ntasks=1 python3 instance.py --partition_id=$partition_id &
done

wait