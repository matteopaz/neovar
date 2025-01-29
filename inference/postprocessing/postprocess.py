import pandas as pd
from pyarrow import parquet as pq
import pyarrow as pa
import pyarrow.compute as pc
import sys
sys.path.append("/home/mpaz/neovar/inference")
from transient_analysis import *
from periodic_analysis import *
import os
from load_data import PartitionDataLoader
from joblib import Parallel, delayed
# from line_profiler import profile 

# gal_parts = list(json.load(open("/home/mpaz/neovar/inference/galactic_partitions.json")))

# done = [int(string.split("_")[1]) for string in os.listdir("/home/mpaz/neovar/inference/out2_filtered") if string.endswith("flag_tbl.csv")]
# to_be_done = [i for i in range(12288) if i not in done]

# SLURM_JOB_ID = os.getenv("SLURM_JOB_ID")
# SLURM_JOB_NAME = os.getenv("SLURM_JOB_NAME")
# SLURM_ARRAY_TASK_N = int(os.getenv("SLURM_ARRAY_TASK_ID"))
# SLURM_ARRAY_TASK_COUNT = int(os.getenv("SLURM_ARRAY_TASK_COUNT"))
# TOTAL_TASKS = len(to_be_done)
# CHUNK_SIZE = TOTAL_TASKS // SLURM_ARRAY_TASK_COUNT
# partitions_to_process = to_be_done[SLURM_ARRAY_TASK_N * CHUNK_SIZE: (SLURM_ARRAY_TASK_N + 1) * CHUNK_SIZE]
partitions_to_process = range(12280,12288)

def centroid(row):
    ra = np.radians(row["ra"])
    dec = np.radians(row["dec"])
    x = np.mean(np.cos(ra) * np.cos(dec))
    y = np.mean(np.sin(ra) * np.cos(dec))
    z = np.mean(np.sin(dec))
    rac = np.arctan2(y, x) * 180 / np.pi
    decc = np.arcsin(z) * 180 / np.pi
    return pd.Series({"ra": np.round(rac, 7), "dec": np.round(decc, 7)})


def process_partition(partition):
    try:
        flags = pd.read_csv(f"/home/mpaz/neovar/inference/out2/partition_{partition}_flag_tbl.csv")
        cids = flags["cluster_id"]
        flags.set_index("cluster_id", inplace=True)
        data = pq.read_table(f"/home/mpaz/neowise-clustering/clustering/out/partition_{partition}_cluster_id_to_data.parquet",
                            filters=pc.is_in(pc.field("cluster_id"), pa.Array.from_pandas(cids))).to_pandas()
    except FileNotFoundError:
        return
    
    data.set_index("cluster_id", inplace=True)

    intersection =  list(set(cids).intersection(set(data.index)))

    flagged_data_subset = data.loc[intersection].reset_index()
    processed_data_pdl = PartitionDataLoader(flagged_data_subset, None)
    processed_data = processed_data_pdl.get_processed_data().set_index("cluster_id")

    processed_data["confidence"] = flags["confidence"]

    transient_inclusion = processed_data.loc[flags["type"] == 1].apply(classify_single_transient, axis=1).apply(lambda x: x == "transient")
    periodic_inclusion = processed_data.loc[flags["type"] == 2].apply(classify_single_periodic, axis=1).apply(lambda x: x == "variable")

    if len(transient_inclusion) == 0:
        inclusion = pd.Series(periodic_inclusion)
    elif len(periodic_inclusion) == 0:
        inclusion = pd.Series(transient_inclusion)
    else:
        inclusion = pd.concat([pd.Series(transient_inclusion), pd.Series(periodic_inclusion)])

    flags["inclusion"] = inclusion
    flags = flags[flags["inclusion"] == True]
    flags.drop(columns=["inclusion"], inplace=True)
    centroids = processed_data.apply(centroid, axis=1)
    flags["ra"] = centroids["ra"]
    flags["dec"] = centroids["dec"]

    flags = flags[["ra", "dec", "type", "confidence"]]

    flags.to_csv(f"/home/mpaz/neovar/inference/out2_filtered/partition_{partition}_flag_tbl.csv") # save the inclusions to the flag tbl


Parallel(n_jobs=-1)(delayed(process_partition)(partition) for partition in partitions_to_process)