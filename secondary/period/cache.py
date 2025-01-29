import pandas as pd
import os
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc
import sys
sys.path.append("/home/mpaz/neovar/inference")
from load_data import PartitionDataLoader
from joblib import Parallel, delayed

SLURM_JOB_ID = os.getenv("SLURM_JOB_ID")
SLURM_JOB_NAME = os.getenv("SLURM_JOB_NAME")
SLURM_ARRAY_TASK_N = int(os.getenv("SLURM_ARRAY_TASK_ID"))
SLURM_ARRAY_TASK_COUNT = int(os.getenv("SLURM_ARRAY_TASK_COUNT"))
TOTAL_TASKS = 12288
partitions_to_process = range(TOTAL_TASKS)[SLURM_ARRAY_TASK_N::SLURM_ARRAY_TASK_COUNT]

rng = np.random.default_rng(0)
partitions_to_process = rng.permutation(partitions_to_process) # make sure high-cadence data is spread out

VALUES_PER_FILE = 512 * 200 * 2 # 512 objs x 200 observations x 2 features

def process_partition(partition_id):
    try:
        flagtbl = pd.read_csv(f"/home/mpaz/neovar/inference/out2_filtered/partition_{partition_id}_flag_tbl.csv")
    except FileNotFoundError:
        print("No flag table for partition", partition_id)
        return
    flagtbl = flagtbl[flagtbl["type"] == 2]
    flagtbl = flagtbl[flagtbl["extragalactic"] == False]

    cids = flagtbl["cluster_id"]
    cid_filter = (pc.is_in(pc.field("cluster_id"), pa.Array.from_pandas(cids)))
    data = pq.read_table(f"/home/mpaz/neowise-clustering/clustering/out/partition_{partition_id}_cluster_id_to_data.parquet",
                        filters=cid_filter).to_pandas()
    
    
    if len(data) == 0:
        print("No phase folds needed for partiton", partition_id)
        return
    
    pdl = PartitionDataLoader(data, len(data))
    filtered, _ = next(iter(pdl))
    filtered.set_index("cluster_id", inplace=True)
    filtered["time"] = filtered["mjd"]
    transform = lambda x: np.arcsinh((x - np.median(x)) / (np.quantile(x, 0.75) - np.quantile(x, 0.25)))
    filtered["mag"] = filtered["w1flux"].apply(transform)
    filtered = filtered[["mag", "time"]]
    return filtered

stack = []
partition_inclusions = []
current_stack_size = 0

def flush():
    global stack
    global partition_inclusions
    global current_stack_size

    total = pd.concat(stack)

    name = "_".join([str(x) for x in partition_inclusions])
    filename = f"/home/mpaz/neovar/secondary/period/cached/{name}.parquet"
    while True:
        filename = f"/home/mpaz/neovar/secondary/period/cached/{name}.parquet"
        if not os.path.exists(filename):
            break

        if name[-1] == "A":
            name = name[:-1] + "B"
        elif name[-1] == "B":
            name = name[:-1] + "C"
        elif name[-1] == "C":
            name = name[:-1] + "D"
        elif name[-1] == "D":
            name = name[:-1] + "E"
        elif name[-1] == "E":
            name = name[:-1] + "F"
        else:
            name += "A"
    
    total.to_parquet(filename, index=True)

    stack = []
    partition_inclusions = []
    current_stack_size = 0

def valuecount(cat):
    return cat["time"].apply(len).sum() * 2

for partition_id in partitions_to_process:
    data = process_partition(partition_id)
    if data is None:
        continue
    values_per_obj = valuecount(data) / len(data)
    print("Processing partition", partition_id, "with", len(data), "objects and", valuecount(data), "values")
    while len(data) > 0:
        partition_inclusions.append(partition_id)
        count = valuecount(data)

        values_left = VALUES_PER_FILE - current_stack_size
        objs_to_add = int(values_left // values_per_obj)

        to_add = data.head(objs_to_add)
        data = data.iloc[objs_to_add:]

        stack.append(to_add)
        current_stack_size += valuecount(to_add)

        if current_stack_size >= 0.9*VALUES_PER_FILE:
            flush()

flush()