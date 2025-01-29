import os
import pandas as pd
import numpy as np
import hpgeom.hpgeom as hpg
from joblib import Parallel, delayed
import json

# SLURM_JOB_ID = os.getenv("SLURM_JOB_ID")
# SLURM_JOB_NAME = os.getenv("SLURM_JOB_NAME")
# SLURM_ARRAY_TASK_N = int(os.getenv("SLURM_ARRAY_TASK_ID"))
# SLURM_ARRAY_TASK_COUNT = int(os.getenv("SLURM_ARRAY_TASK_COUNT"))
# TOTAL_TASKS = 12300
# CHUNK_SIZE = TOTAL_TASKS // SLURM_ARRAY_TASK_COUNT
# partitions_to_process = range(SLURM_ARRAY_TASK_N * CHUNK_SIZE, (SLURM_ARRAY_TASK_N + 1) * CHUNK_SIZE)
partitions_to_process = range(12288)

radius = 2 # in arcseconds
catalog_priority = ["gaia-general", "gaia-cepheids", "gaia-ebs", "gaia-lpvs", 
                    "gaia-oscillators", "gaia-rotators", "gaia-rrlyrae", "chen-ztf", "petrosky",
                    "chen-neowise", "assef-agn", "spicy", "gaia-sts"]
typemap, typemap_types = json.load(open("/home/mpaz/neovar/secondary/xmatch/class_merging.json"))


catalogs = [pd.read_csv(f"/home/mpaz/neovar/secondary/catalogs/{f}") for f in os.listdir("/home/mpaz/neovar/secondary/catalogs/") if f.endswith(".csv")]
catalog = pd.concat(catalogs)

catalog["partition"] = hpg.angle_to_pixel(32, catalog["ra"], catalog["dec"])
catalog = catalog.loc[catalog["partition"].isin(partitions_to_process)]
grouped = catalog.groupby("partition")

def get_type(types):
    if "EA" in types.values:
        return typemap["EA"]
    elif "EW" in types.values:
        return typemap["EW"]

    pre_type = types.iloc[0]
    if pre_type in typemap_types:
        return typemap[types.iloc[0]]
    else:
        print(f"Warning: type {pre_type} not found in types")
        return types.iloc[0]

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

def getpd(pds):
    pds = np.array(pds)
    pds = pds[~np.isnan(pds)]
    return pds[0] if len(pds) > 0 else np.nan

def xmatch_partition(partition, group):
    print(partition)
    try:
        flag_tbl = pd.read_csv(f"/home/mpaz/neovar/inference/flagtbls/partition_{partition}_flag_tbl.csv").reset_index() # Todo
    except FileNotFoundError:
        return 
    get_dists = get_dists_gen(flag_tbl)
    group["dists"] = group.apply(get_dists, axis=1)
    int_closest = group["dists"].apply(np.argmin)
    group["dist"] = group["dists"].apply(np.min)
    group["cluster_id"] = flag_tbl.loc[int_closest, "cluster_id"].to_numpy()
    group["cluster_id"] = group["cluster_id"].astype("Int64")
    group = group.loc[group["dist"] < radius]
    group = group.sort_values(by="catalog", key=lambda catalog: catalog.apply(lambda x: catalog_priority.index(x)))
    objects = group.groupby("cluster_id")

    # Columns for compacted table
    catalogs = objects["catalog"].apply(lambda x: x.iloc[0])
    types = objects["type"].apply(get_type)
    ra = objects["ra"].apply(lambda x: x.iloc[0])
    dec = objects["dec"].apply(lambda x: x.iloc[0])
    dist = objects["dist"].apply(lambda x: x.iloc[0])
    cluster_id = objects["cluster_id"].apply(lambda x: x.iloc[0])

    new = pd.DataFrame({"catalog": catalogs, "type": types, "ra": ra, "dec": dec, "dist": dist, "cluster_id": cluster_id})
    new.set_index("cluster_id", inplace=True)
    return new

tables = Parallel(n_jobs=28)(delayed(xmatch_partition)(partition, group) for partition, group in grouped)
result = pd.concat(tables)

result.to_csv(f"/home/mpaz/neovar/secondary/xmatch/xmatched.csv", index=True)
    