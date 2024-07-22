import os
import pandas as pd
import re

def unfinished_partitions():
    directory = os.fsencode("/home/mpaz/neowise-clustering/clustering/out/")
    partition_ids = set(list(range(12288)))
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        # check if matches regex
        if filename.endswith("cntr_to_cluster_id.csv"):
            parts = filename.split("_")
            pid = int(parts[1])
            if pid in partition_ids:
                partition_ids.remove(pid)
    return partition_ids

def sizeof(partition_id):
    path = lambda yr: "/stage/irsa-data-parquet10/wise/neowiser/p1bs_psd/healpix_k5/" + f"year{yr}_skinny/row-counts-per-partition.csv"
    nrows = 0
    for yr in range(1,11):
        # Access the row counts for the year
        df = pd.read_csv(path(yr))
        # Get the row count for the partition
        nrows += df[df["healpix_k5"] == partition_id]["nrows"].values[0]
    return nrows

unf = unfinished_partitions()
l = [(pid, sizeof(pid)) for pid in unf]
print(l)
print(sum([x[1] for x in l]))