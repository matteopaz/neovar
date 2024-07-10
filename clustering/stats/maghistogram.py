import pyarrow as pa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from joblib import Parallel, delayed
from time import perf_counter
import os

SLURM_ARRAY_TASK_ID = os.getenv("SLURM_ARRAY_TASK_ID")
SLURM_TASK_JOB_CT = os.getenv("SLURM_ARRAY_TASK_COUNT")

size = int(12288 // int(SLURM_TASK_JOB_CT))
start = int(SLURM_ARRAY_TASK_ID) * size
end = start + size

basepath = "/home/mpaz/neowise-clustering/clustering/out/"
filename = lambda partition_id: f"partition_{partition_id}_cluster_id_to_data.parquet"
path = lambda partition_id: basepath + filename(partition_id)

def fluxtomag(x: np.ndarray):
    negative_indices = x < 0
    y = x.copy()
    y[negative_indices] = np.nan
    return 20.752 - 2.5 * np.log10(y)

def get_magnitudes(partition_id):
    try:
        table = pd.read_parquet(path(partition_id), columns=["w1flux"])
    except:
        return [-100]
    # each row is a list of magnitudes. take the median of each list
    mags = table["w1flux"].apply(fluxtomag)
    medians = mags.apply(np.nanmedian).values
    if np.isnan(medians).all():
        return [-100]
    
    return medians

chunk_magnitudes = Parallel(n_jobs=-1)(delayed(get_magnitudes)(i) for i in range(start, end))
magnitudes = [item for sublist in chunk_magnitudes for item in sublist]

json.dump(magnitudes, open(f"/home/mpaz/neowise-clustering/clustering/magnitudes_{SLURM_ARRAY_TASK_ID}.json", "w"))

# create histogram of magnitudes
# plt.hist(magnitudes, bins=100)
# plt.xlabel("Magnitude")
# plt.ylabel("Number of sources")
# plt.title("Histogram of magnitudes of sources in the sky")
# plt.savefig("/home/mpaz/neowise-clustering/clustering/maghistogram.png")