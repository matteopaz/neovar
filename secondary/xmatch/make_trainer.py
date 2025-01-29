import sys
sys.path.append('../inference/')
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import hpgeom.hpgeom as hpg
import json
# from astropy import units as u
# from astroquery.xmatch import XMatch
# from astropy.table import Table
import os

SLURM_JOB_ID = os.getenv("SLURM_JOB_ID")
SLURM_JOB_NAME = os.getenv("SLURM_JOB_NAME")
SLURM_ARRAY_TASK_N = int(os.getenv("SLURM_ARRAY_TASK_ID"))
SLURM_ARRAY_TASK_COUNT = int(os.getenv("SLURM_ARRAY_TASK_COUNT"))
TOTAL_TASKS = 12288
CHUNK_SIZE = TOTAL_TASKS // SLURM_ARRAY_TASK_COUNT
partitions_to_process = range(SLURM_ARRAY_TASK_N * CHUNK_SIZE, (SLURM_ARRAY_TASK_N + 1) * CHUNK_SIZE)

catalog = pd.read_csv("/home/mpaz/neovar/secondary/xmatch/xmatched.csv").set_index("cluster_id")
catalog["partition"] = hpg.angle_to_pixel(32, catalog["ra"], catalog["dec"])
catalog = catalog[catalog["partition"].isin(partitions_to_process)]

partition_groups = catalog.groupby("partition")

typemap, _ = json.load(open("/home/mpaz/neovar/secondary/xmatch/class_merging.json"))

data_list = []

keeptypes = ["ea", "ew", "lpv", "agn", "rot", "rr", "cep", "rscvn", "yso", "acv"] # IMPORTANT

def process(group):
    partition=group["partition"].iloc[0]
    group["type"] = group["type"].apply(lambda x: x if x in keeptypes else pd.NA)
    group.dropna(subset=["type"], inplace=True)
    cids = pa.Array.from_pandas(group.index).cast(pa.int64())
    data = pq.read_table("/home/mpaz/neowise-clustering/clustering/out/partition_{}_cluster_id_to_data.parquet".format(partition),
                         filters=pa.compute.is_in(pa.compute.field("cluster_id"), cids)).to_pandas().set_index("cluster_id")
    
    ftb = pd.read_csv("/home/mpaz/neovar/inference/flagtbls/partition_{}_flag_tbl.csv".format(partition)).set_index("cluster_id")

    ftb = ftb.drop(columns=["ra", "dec", "type"])
    group = group.drop(columns=["ra", "dec", "partition"])
    
    group = group.join(data, how="left")
    group = group.join(ftb, how="left")

    group.dropna(subset=["type"], inplace=True)
    data_list.append(group)

partition_groups.apply(process)
data = pd.concat(data_list)
print(len(catalog))
print(len(data))

data.to_parquet(f"/home/mpaz/neovar/secondary/data/training_data_{SLURM_ARRAY_TASK_N}.parquet")