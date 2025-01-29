import pandas as pd
from pyarrow import parquet as pq
import pyarrow as pa
import pyarrow.compute as pc
import sys
sys.path.append("/home/mpaz/neovar/inference")
from transient_analysis import *
from periodic_analysis import *
import os
from lib import get_centroid
from load_data import PartitionDataLoader
import json
from joblib import Parallel, delayed
# from line_profiler import profile 

# gal_parts = list(json.load(open("/home/mpaz/neovar/inference/galactic_partitions.json")))

# SLURM_JOB_ID = os.getenv("SLURM_JOB_ID")
# SLURM_JOB_NAME = os.getenv("SLURM_JOB_NAME")
# SLURM_ARRAY_TASK_N = int(os.getenv("SLURM_ARRAY_TASK_ID"))
# SLURM_ARRAY_TASK_COUNT = int(os.getenv("SLURM_ARRAY_TASK_COUNT"))
# TOTAL_TASKS = 12300
# CHUNK_SIZE = TOTAL_TASKS // SLURM_ARRAY_TASK_COUNT
# partitions_to_process = range(SLURM_ARRAY_TASK_N * CHUNK_SIZE, (SLURM_ARRAY_TASK_N + 1) * CHUNK_SIZE)
partitions_to_process = [11602, 11603, 11604, 11605, 11607, 11608, 11606, 11609, 11610, 11611, 11612, 11613, 11614, 11615, 11616, 11617, 11618, 11619, 11620, 11621, 11622, 11623, 11624, 11625, 11626, 11627, 11628, 11629, 11630, 11631, 11632, 11633, 11634, 11635, 11636, 11639, 11640, 11641, 11637, 11643, 11645, 11646, 11638, 11642, 11644, 11647, 11648, 11649, 11650, 11651, 11652, 11653, 11654, 11655, 11656, 11657, 11658, 11659, 11660, 11661, 11662, 11663, 11664, 11665, 11666, 11667, 11668, 11669, 11670, 11671, 11672, 11673, 11674, 11675, 11676, 11677, 11678, 11679, 11680, 11681, 11682, 11683, 11684, 11685, 11686, 11687, 11688, 11689, 11690, 11691, 11692, 11693, 11694, 11695, 11696, 11697, 11698, 11699, 11700, 11701, 11702, 11703, 11704, 11705, 11706, 11707, 11708, 11709, 11710, 11711]

def cluster_id_centroid(row):
    ra = row["ra"].apply(lambda x: x / 180 * np.pi)
    dec = row["dec"].apply(lambda x: x / 180 * np.pi)
    x = (ra.apply(np.cos) * dec.apply(np.cos)).apply(np.mean)
    y = (ra.apply(np.sin) * dec.apply(np.cos)).apply(np.mean)
    z = (dec.apply(np.sin)).apply(np.mean)
    rac = np.arctan2(y, x) * 180 / np.pi
    decc = np.arcsin(z) * 180 / np.pi

    return pd.Series({"ra": np.round(rac, 6), "dec": np.round(decc, 5)})

def cid_to_ra(cid):
    cid = int(cid)
    cid = cid & int("0"*16 + "1"*24 + "0"*24, 2)

    ra = (cid >> 24) * 0.0001
    return round(ra, 4)

def cid_to_dec(cid):
    cid = int(cid)
    cid = cid & int("0"*40 + "1"*24, 2)
    dec = 90.0 - (cid * 0.0001)
    return round(dec, 4)

def fix_centroids(partition):
    try:
        flags = pd.read_csv(f"/home/mpaz/neovar/inference/map_out/partition_{partition}_flag_tbl.csv")
        cids = flags["cluster_id"]
        data = pq.read_table(f"/home/mpaz/neowise-clustering/clustering/out/partition_{partition}_cluster_id_to_data.parquet",
                            filters=pc.is_in(pc.field("cluster_id"), pa.Array.from_pandas(cids))).to_pandas()
    except FileNotFoundError:
        return
    
    flags.set_index("cluster_id", inplace=True)
    data.set_index("cluster_id", inplace=True)
    centroids = data[["ra", "dec"]].apply(lambda row: get_centroid(row["ra"], row["dec"]), axis=1)
    flags["ra"] = centroids.apply(lambda x: np.round(x[0], 6) % 360)
    flags["dec"] = centroids.apply(lambda x: np.round(x[1], 6))
    
    othercols = flags.columns.difference(["ra", "dec", "cluster_id"])
    flags.reset_index(inplace=True)
    flags = flags[["cluster_id", "ra", "dec"] + list(othercols)]
    flags.to_csv(f"/home/mpaz/neovar/inference/map_out/partition_{partition}_flag_tbl.csv", index=False)
    os.remove(f"/home/mpaz/neovar/inference/map_out/partition_{partition}_flag_tblcent.csv")

def filter_to_inclusion(partition):
    try:
        flags = pd.read_csv(f"/home/mpaz/neovar/inference/map_out/partition_{partition}_flag_tbl.csv")
        flags.set_index("cluster_id", inplace=True)
    except FileNotFoundError:
        return
    
    detections = flags[flags["inclusion"]==True]
    detections = detections[["ra", "dec", "type", "confidence"]]
    detections.to_csv(f"/home/mpaz/neovar/inference/map_out/partition_{partition}_flag_tbl.csv")


def process_partition(partition):
    try:
        flags = pd.read_csv(f"/home/mpaz/neovar/inference/out2/partition_{partition}_flag_tbl.csv")
        cids = flags["cluster_id"]
        flags.set_index("cluster_id", inplace=True)
        original = pq.read_table(f"/home/mpaz/neowise-clustering/clustering/out/partition_{partition}_cluster_id_to_data.parquet",
                            filters=pc.is_in(pc.field("cluster_id"), pa.Array.from_pandas(cids))).to_pandas()
    except FileNotFoundError:
        return
    
    data = original.copy()
    data.set_index("cluster_id", inplace=True)

    flagged_data_subset = data.loc[cids].reset_index()
    processed_data_pdl = PartitionDataLoader(flagged_data_subset, None)
    processed_data = processed_data_pdl.get_processed_data().set_index("cluster_id")

    try:
        transient_inclusion = processed_data.loc[flags["type"] == 1].apply(classify_single_transient, axis=1).apply(lambda x: x == "transient")
        periodic_inclusion = processed_data.loc[flags["type"] == 2].apply(classify_single_periodic, axis=1).apply(lambda x: x == "variable")

        # confident = flags["confidence"] > 0.99 # GALAXTIC

        # transient_inclusion[~confident] = pd.NA # GALACTIC
        # periodic_inclusion[~confident] = pd.NA # GALACTIC
        
        if len(transient_inclusion) == 0:
            inclusion = pd.Series(periodic_inclusion)
        elif len(periodic_inclusion) == 0:
            inclusion = pd.Series(transient_inclusion)
        else:
            inclusion = pd.concat([pd.Series(transient_inclusion), pd.Series(periodic_inclusion)])
    except Exception as e:
        print(f"Error in partition {partition}: {e}")

    flags["inclusion"] = inclusion

    flags.to_csv(f"/home/mpaz/neovar/inference/out2/partition_{partition}_flag_tbl.csv") # save the inclusions to the flag tbl

    # data["filter_mask"] = processed_data["filter_mask"]

    # data.reset_index(inplace=True)
    # matching = (data[original.keys()] == original).all(axis=None)

    # if matching:
    #     data.to_parquet(f"/home/mpaz/neowise-clustering/clustering/out/partition_{partition}_cluster_id_to_data.parquet")
    # else:
    #     raise Warning(f"Partition {partition} does not match")
    

# for partition in partitions_to_process:
#     # fix_centroids(partition)
#     print(partition)
#     process_partition(partition)
#     # filter_to_inclusion(partition)

Parallel(n_jobs=-6)(delayed(filter_to_inclusion)(partition) for partition in partitions_to_process)