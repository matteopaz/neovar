from run_partition_cached import run_partition
import time
import tqdm
from joblib import Parallel, delayed
import os

# cache_dir = "/home/mpaz/neowise-clustering/clustering/inference_ready/"

# partitions_to_do = set([int(f.split("_")[0][1:]) for f in os.listdir(cache_dir) if f.endswith("_tensor.pt")])
# partitions_done = set([int(f.split("_")[1]) for f in os.listdir("map_out") if f.endswith("_flag_tbl.csv")])

# partitions_to_do = list(partitions_to_do - partitions_done)
# partitions_to_do = [1]
done = set([int(f.split("_")[1]) for f in os.listdir("out2") if f.endswith("_flag_tbl.csv")])

ptd = list(range(12288))
partitions_to_do = [ptd[523*i % len(ptd)] for i in range(len(ptd))] # shuffles for an accurate time estimate
partitions_to_do = list(set(partitions_to_do) - done)

start_t = time.time()

# for i, partition_id in enumerate(partitions_to_do):

def process(partition_id):
    # print(f"Partition {partition_id}")
    flag_tbl = run_partition(partition_id)
    if flag_tbl is None:
        return

    flag_tbl.sort_values("confidence", inplace=True, ascending=False)
    # now_t = time.time()
    flag_tbl.to_csv(f"out2/partition_{partition_id}_flag_tbl.csv", index=False)
    
    # if i % 128 == 0:
    #     print("Partition ", partition_id, " of ", len(partitions_to_do))
    #     print("Est total time left: ", ((now_t - start_t) / (partition_id + 1) * (len(partitions_to_do) - i)) / 3600, " hours")

Parallel(n_jobs=12)(delayed(process)(partition_id) for partition_id in tqdm.tqdm(partitions_to_do))

