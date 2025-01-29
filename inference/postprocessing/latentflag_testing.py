import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

partitions_to_process = range(10)

catalog = pd.concat([pd.read_csv("/home/mpaz/neovar/inference/map_out/partition_{}_flag_tbl.csv".format(i)) for i in partitions_to_process]).set_index("cluster_id")
catalog = catalog[catalog["type"] == 1]

cids = pa.array(catalog.index, type=pa.int64())
readpq = lambda f: pq.read_table("/home/mpaz/neowise-clustering/clustering/out/partition_{}_cluster_id_to_data.parquet".format(f),
                                filters=pc.is_in(pc.field("cluster_id"), cids)).to_pandas()
data = pd.concat([readpq(i) for i in partitions_to_process]).set_index("cluster_id")

def get_epochs(row):
    t = row["mjd"]
    dt = np.concatenate((np.zeros(1), np.diff(t)))
    epochskip = np.where(dt > 50)[0]
    epochlen = np.diff(np.concatenate((epochskip, [len(t)])))

    te = np.split(t, epochskip)
    fe = np.split(row["w1flux"], epochskip)
    se = np.split(row["w1sigflux"], epochskip)
    return te, fe, se

def zigscore(row):
    row["w1flux"] = (row["w1flux"] - np.nanmedian(row["w1flux"])) / (np.nanquantile(row["w1flux"], 0.9) - np.nanquantile(row["w1flux"], 0.1))
    epochs_t, epochs_f, epochs_s = get_epochs(row)

    epoch_f = np.array([np.nanmedian(e) for e in epochs_f])
    dp = np.diff(epoch_f)
    pattern = np.zeros_like(dp)
    pattern[dp > 0] = 1
    pattern[dp < 0] = -1
    up_triangles = 0
    for i in range(len(pattern) - 1):
        if pattern[i] == 1 and pattern[i+1] == -1:
            up_triangles += 1

    score = up_triangles / (len(pattern) // 2)
    return score

def varscore(row):
    row["w1flux"] = (row["w1flux"] - np.nanmedian(row["w1flux"])) / (np.nanquantile(row["w1flux"], 0.9) - np.nanquantile(row["w1flux"], 0.1))
    epochs_t, epochs_f, epochs_s = get_epochs(row)
    north_scan_dir = np.array([np.nanmedian(e) for e in epochs_f[0::2]])
    south_scan_dir = np.array([np.nanmedian(e) for e in epochs_f[1::2]])
    overall = np.array([np.nanmedian(e) for e in epochs_f])

    northvar = np.nanvar(north_scan_dir)
    southvar = np.nanvar(south_scan_dir)

    score = min(northvar, southvar) / np.nanvar(overall)
    return score
    

data["zig"] = data.apply(zigscore, axis=1)
data["var"] = data.apply(varscore, axis=1)
catalog["zig"] = data["zig"]
catalog["var"] = data["var"]
catalog["flag"] = (catalog["zig"] > 0.8) & (catalog["var"] < 0.3) | (catalog["var"] < 0.1)

catalog.sort_values("zig", ascending=False, inplace=True)
catalog[["ra", "dec", "zig", "var", "flag"]].to_csv("latentscore.csv")