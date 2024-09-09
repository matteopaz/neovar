from run_partition_cached import run_partition
import numpy as np
import time

partitions_to_do = range(12888)
# partitions_to_do = [8977]

start_t = time.time()

for partition_id in partitions_to_do:
    print(f"Partition {partition_id}")
    flag_tbl = run_partition(partition_id)
    now_t = time.time()
    print("Est total time: ", (now_t - start_t) / (partition_id + 1) * len(partitions_to_do))
    flag_tbl.to_csv(f"out/partition_{partition_id}_flag_tbl.csv", index=False)
