import os
import pandas as pd
import numpy as np
import hpgeom.hpgeom as hpg
from joblib import Parallel, delayed
import tqdm

SLURM_JOB_ID = os.getenv("SLURM_JOB_ID")
SLURM_JOB_NAME = os.getenv("SLURM_JOB_NAME")
SLURM_ARRAY_TASK_N = int(os.getenv("SLURM_ARRAY_TASK_ID"))
SLURM_ARRAY_TASK_COUNT = int(os.getenv("SLURM_ARRAY_TASK_COUNT"))
TOTAL_TASKS = 12288
CHUNK_SIZE = TOTAL_TASKS // SLURM_ARRAY_TASK_COUNT
partitions_to_process = range(SLURM_ARRAY_TASK_N * CHUNK_SIZE, (SLURM_ARRAY_TASK_N + 1) * CHUNK_SIZE)
# partitions_to_process = range(1)

radius = 2 # in arcseconds

gaiagal = pd.read_csv("/home/mpaz/neovar/inference/postprocessing/tables/purer_galaxy_sub-sample.csv")
gaiaqso = pd.read_csv("/home/mpaz/neovar/inference/postprocessing/tables/purer_quasar_sub-sample.csv")
# allwise = pd.read_csv("/home/mpaz/neovar/inference/postprocessing/tables/allwise.csv")

gaiagal["partition"] = hpg.angle_to_pixel(32, gaiagal["ra"], gaiagal["dec"])
gaiagal = gaiagal.loc[gaiagal["partition"].isin(partitions_to_process)]
gaiagal_grouped = gaiagal.groupby("partition")

gaiaqso["partition"] = hpg.angle_to_pixel(32, gaiaqso["ra"], gaiaqso["dec"])
gaiaqso = gaiaqso.loc[gaiaqso["partition"].isin(partitions_to_process)]
gaiaqso_grouped = gaiaqso.groupby("partition")

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in arcseconds between two points in RA and Dec
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    deg = np.degrees(c)
    return deg * 3600

def get_dists_gen(flag_tbl):
    return lambda row: haversine(row["ra"], row["dec"], flag_tbl["ra"], flag_tbl["dec"]).values

tables = []

def xmatch_partition(partition, group, getcols=[]):
    # print(partition)
    try:
        flag_tbl = pd.read_csv(f"/home/mpaz/neovar/inference/out2_filtered/partition_{partition}_flag_tbl.csv")
    except FileNotFoundError:
        return 
    
    get_dists = get_dists_gen(flag_tbl)
    group["dists"] = group.apply(get_dists, axis=1)
    int_closest = group["dists"].apply(np.argmin)
    group["dist"] = group["dists"].apply(np.min)
    group["cluster_id"] = flag_tbl.loc[int_closest, "cluster_id"].to_numpy()
    group["cluster_id"] = group["cluster_id"].astype("Int64")
    group = group.loc[group["dist"] < radius]

    sorted = group.sort_values("dist")
    same_obj = sorted.groupby("cluster_id")

    cluster_id = same_obj["cluster_id"].apply(lambda x: x.iloc[0])
    n_in_radius = same_obj["cluster_id"].apply(len)
    ra = same_obj["ra"].apply(lambda x: x.iloc[0])
    dec = same_obj["dec"].apply(lambda x: x.iloc[0])
    dist = same_obj["dist"].apply(lambda x: x.iloc[0])
    retrieved_columns = []
    for col in getcols:
        retrieved_columns.append(same_obj[col].apply(lambda x: x.iloc[0]))

    xmatched = pd.DataFrame({"cluster_id": cluster_id, "n_in_radius": n_in_radius, "dist": dist, **{col: retrieved_columns[i] for i, col in enumerate(getcols)}})
    xmatched.set_index("cluster_id", inplace=True)
    return partition, xmatched

qso = Parallel(n_jobs=-8)(delayed(xmatch_partition)(partition, group) for partition, group in gaiaqso_grouped)
gal = Parallel(n_jobs=-8)(delayed(xmatch_partition)(partition, group) for partition, group in gaiagal_grouped)

flagtbls = {}
for i in partitions_to_process:
    try:
        flagtbls[i] = pd.read_csv(f"/home/mpaz/neovar/inference/out2_filtered/partition_{i}_flag_tbl.csv").set_index("cluster_id")
    except FileNotFoundError:
        pass


for qsoi in qso:
    if qsoi is None:
        continue
    pid, xmatched = qsoi
    flagtbl = flagtbls[pid]
    flagtbl["qso"] = False
    flagtbl.loc[xmatched.index, "qso"] = True

for gali in gal:
    if gali is None:
        continue
    pid, xmatched = gali
    flagtbl = flagtbls[pid]
    flagtbl["gal"] = False
    flagtbl.loc[xmatched.index, "gal"] = True

for pid, flagtbl in flagtbls.items():
    if "qso" not in flagtbl.columns:
        flagtbl["qso"] = False
    if "gal" not in flagtbl.columns:
        flagtbl["gal"] = False

    flagtbl["extragalactic"] = (flagtbl["qso"] == True) | (flagtbl["gal"] == True)
    flagtbl.drop(columns=["qso", "gal"], inplace=True)
    flagtbl.to_csv(f"/home/mpaz/neovar/inference/out2_filtered/partition_{pid}_flag_tbl.csv")
