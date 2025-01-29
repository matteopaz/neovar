import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import os
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
import pickle as pkl

# SLURM_JOB_ID = os.getenv("SLURM_JOB_ID")
# SLURM_JOB_NAME = os.getenv("SLURM_JOB_NAME")
# SLURM_ARRAY_TASK_N = int(os.getenv("SLURM_ARRAY_TASK_ID"))
# SLURM_ARRAY_TASK_COUNT = int(os.getenv("SLURM_ARRAY_TASK_COUNT"))
# TOTAL_TASKS = 12300
# CHUNK_SIZE = TOTAL_TASKS // SLURM_ARRAY_TASK_COUNT
# partitions_to_process = range(SLURM_ARRAY_TASK_N * CHUNK_SIZE, (SLURM_ARRAY_TASK_N + 1) * CHUNK_SIZE)

partitions_to_process = range(10)

catalog = pd.concat([pd.read_csv("/home/mpaz/neovar/inference/map_out/partition_{}_flag_tbl.csv".format(i)) for i in partitions_to_process]).set_index("cluster_id")

cids = pa.array(catalog.index, type=pa.int64())
readpq = lambda f: pq.read_table("/home/mpaz/neowise-clustering/clustering/out/partition_{}_cluster_id_to_data.parquet".format(f),
                                filters=pc.is_in(pc.field("cluster_id"), cids)).to_pandas()
data = pd.concat([readpq(i) for i in partitions_to_process]).set_index("cluster_id")

def blendscore(ra, dec):
    radec = np.array([ra, dec]).T
    ctr = np.mean(radec, axis=0)
    radec = radec - ctr
    radec = radec * 3600

    km = KMeans(n_clusters=2)
    km.fit(radec)

    maxra = np.quantile(radec[:, 0], 0.9)
    minra = np.quantile(radec[:, 0], 0.1)
    maxdec = np.quantile(radec[:, 1], 0.9)
    mindec = np.quantile(radec[:, 1], 0.1)

    span = np.sqrt((maxra - minra) ** 2 + (maxdec - mindec) ** 2)

    dist = np.linalg.norm(km.cluster_centers_[0] - km.cluster_centers_[1])
    # print(span)
    # print(dist)
    print(dist/span)

    score = (km.score(radec))
    if dist > span:
        print(ctr)
    
    return score

data["blendscore"] = data.apply(lambda x: blendscore(x["ra"], x["dec"]), axis=1)
catalog["blendscore"] = data["blendscore"]
catalog["blendflag"] = catalog["blendscore"] > 1.2
catalog.sort_values("blendscore", ascending=False, inplace=True)
catalog[["ra", "dec", "blendscore", "blendflag", "confidence"]].to_csv("blendscore.csv")