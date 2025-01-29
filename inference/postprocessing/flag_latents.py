import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import os
from joblib import Parallel, delayed

# SLURM_JOB_ID = os.getenv("SLURM_JOB_ID")
# SLURM_JOB_NAME = os.getenv("SLURM_JOB_NAME")
# SLURM_ARRAY_TASK_N = int(os.getenv("SLURM_ARRAY_TASK_ID"))
# SLURM_ARRAY_TASK_COUNT = int(os.getenv("SLURM_ARRAY_TASK_COUNT"))
# TOTAL_TASKS = 12300
# CHUNK_SIZE = TOTAL_TASKS // SLURM_ARRAY_TASK_COUNT
# partitions_to_process = range(SLURM_ARRAY_TASK_N * CHUNK_SIZE, (SLURM_ARRAY_TASK_N + 1) * CHUNK_SIZE)
partitions_to_process = [11602, 11603, 11604, 11605, 11607, 11608, 11606, 11609, 11610, 11611, 11612, 11613, 11614, 11615, 11616, 11617, 11618, 11619, 11620, 11621, 11622, 11623, 11624, 11625, 11626, 11627, 11628, 11629, 11630, 11631, 11632, 11633, 11634, 11635, 11636, 11639, 11640, 11641, 11637, 11643, 11645, 11646, 11638, 11642, 11644, 11647, 11648, 11649, 11650, 11651, 11652, 11653, 11654, 11655, 11656, 11657, 11658, 11659, 11660, 11661, 11662, 11663, 11664, 11665, 11666, 11667, 11668, 11669, 11670, 11671, 11672, 11673, 11674, 11675, 11676, 11677, 11678, 11679, 11680, 11681, 11682, 11683, 11684, 11685, 11686, 11687, 11688, 11689, 11690, 11691, 11692, 11693, 11694, 11695, 11696, 11697, 11698, 11699, 11700, 11701, 11702, 11703, 11704, 11705, 11706, 11707, 11708, 11709, 11710, 11711]

def get_epochs(row):
    t = row["mjd"]
    dt = np.concatenate((np.zeros(1), np.diff(t)))
    epochskip = np.where(dt > 50)[0]

    te = np.split(t, epochskip)
    fe = np.split(row["w1flux"], epochskip)
    se = np.split(row["w1sigflux"], epochskip)
    return te, fe, se

def zigscore(row):
    row["w1flux"] = (row["w1flux"] - np.nanmedian(row["w1flux"])) / (np.nanquantile(row["w1flux"], 0.9) - np.nanquantile(row["w1flux"], 0.1))
    epochs_t, epochs_f, epochs_s = get_epochs(row)

    epoch_f = np.array([np.nanmedian(e) for e in epochs_f])
    if np.isnan(epoch_f).any():
        return 0

    if len(epochs_f) < 2:
        return 0

    dp = np.diff(epoch_f)
    pattern = np.zeros_like(dp)
    pattern[dp > 0] = 1
    pattern[dp < 0] = -1

    up_triangles = 0

    for i in range(len(pattern) - 1):
        if pattern[i] == 1 and pattern[i+1] == -1:
            up_triangles += 1

    if len(pattern) < 2:
        return 0

    score = up_triangles / (len(pattern) // 2)
    return score

def varscore(row):
    row["w1flux"] = (row["w1flux"] - np.nanmedian(row["w1flux"])) / (np.nanquantile(row["w1flux"], 0.9) - np.nanquantile(row["w1flux"], 0.1))
    epochs_t, epochs_f, epochs_s = get_epochs(row)

    if len(epochs_f) < 2:
        return 1
    
    north_scan_dir = np.array([np.nanmedian(e) for e in epochs_f[0::2]])

    south_scan_dir = np.array([np.nanmedian(e) for e in epochs_f[1::2]])
    overall = np.array([np.nanmedian(e) for e in epochs_f])   

    if np.isnan(overall).any():
        return 1

    if np.isnan(north_scan_dir).any():
        row["w1flux"] = row["w1flux"] + 1

    

    northvar = np.nanvar(north_scan_dir)
    southvar = np.nanvar(south_scan_dir)

    score = min(northvar, southvar) / np.nanvar(overall)
    return score
    
def process(partition):
    try:
        catalog = pd.read_csv("/home/mpaz/neovar/inference/map_out/partition_{}_flag_tbl.csv".format(partition)).set_index("cluster_id")  
    except:
        return  
    cids = pa.array(catalog[catalog["type"] == 1].index, type=pa.int64())
    if len(cids) == 0:
        return
    data = pq.read_table("/home/mpaz/neowise-clustering/clustering/out/partition_{}_cluster_id_to_data.parquet".format(partition),
                                    filters=pc.is_in(pc.field("cluster_id"), cids)).to_pandas().set_index("cluster_id")
    
    data["zig"] = data.apply(zigscore, axis=1)
    data["var"] = data.apply(varscore, axis=1)
    data["latent_flag"] = ((data["zig"] > 0.8) & (data["var"] < 0.3)) | (data["var"] < 0.1)
    data["latent_flag"] = data["latent_flag"].astype("boolean")
    data[data["latent_flag"] == False] = pd.NA
    catalog["latent_flag"] = data['latent_flag']
    catalog.to_csv("/home/mpaz/neovar/inference/map_out/partition_{}_flag_tbl.csv".format(partition))

    
Parallel(n_jobs=-6)(delayed(process)(partition) for partition in partitions_to_process)