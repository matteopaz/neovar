from func import find_clusters_one_partition
from datetime import datetime
from time import perf_counter
import pyarrow.parquet as pq
import pyarrow
import pandas as pd
import pyarrow.dataset
import os
import argparse
from math import ceil, log2

MAX_ROWS_TO_LOAD = 8 * 10**6 # 7 Million

def log(*args):
    with open(PATH_TO_OUTFILE, "a") as f:
        print(*args, file=f)

def sizeof(partition_id):
    path = lambda yr: "/stage/irsa-data-parquet10/wise/neowiser/p1bs_psd/healpix_k5/" + f"year{yr}_skinny/row-counts-per-partition.csv"
    nrows = 0
    for yr in range(1,11):
        # Access the row counts for the year
        df = pd.read_csv(path(yr))
        # Get the row count for the partition
        nrows += df[df["healpix_k5"] == partition_id]["nrows"].values[0]
    return nrows

try:
    SLURM_JOB_ID = os.getenv("SLURM_JOB_ID")
    SLURM_JOB_NAME = os.getenv("SLURM_JOB_NAME")
    SLURM_ARRAY_TASK_N = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    SLURM_ARRAY_TASK_COUNT = int(os.getenv("SLURM_ARRAY_TASK_COUNT"))
    TOTAL_TASKS = 12288
    CHUNK_SIZE = TOTAL_TASKS // SLURM_ARRAY_TASK_COUNT

    OUTFILE_NAME = SLURM_JOB_NAME + "_" + str(SLURM_ARRAY_TASK_N) + ".out"
    PATH_TO_OUTFILE = "/home/mpaz/neowise-clustering/clustering/logs/" + OUTFILE_NAME

    start_idx = SLURM_ARRAY_TASK_N * CHUNK_SIZE
    end_idx = start_idx + CHUNK_SIZE
except:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition-id", type=int, default=-1)
    args = parser.parse_args()
    start_idx = args.partition_id
    end_idx = start_idx + 1

    log=print

base_path = "/stage/irsa-data-parquet10/wise/neowiser/p1bs_psd/healpix_k5/" 
year_path = "year<N>_skinny/neowiser-healpix_k5-year<N>_skinny.parquet/_metadata"
neowise_path = lambda year: base_path + year_path.replace("<N>", str(year))

year_datasets = [
pyarrow.dataset.parquet_dataset(neowise_path(year), partitioning="hive") for year in range(1, 11)
]
neowise_ds = pyarrow.dataset.dataset(year_datasets)


for partition_k_pixel_id in range(start_idx, end_idx):
    log("Beginning clustering on partition {} at {}".format(partition_k_pixel_id, datetime.now()))

    nrows = sizeof(partition_k_pixel_id)
    subdivisions = ceil(0.5 * log2(nrows / MAX_ROWS_TO_LOAD)) # Log 4 of the number of rows
    iter_k = 5 + subdivisions
    log("Partition {} has {} rows. Using {} subdivisions - iter_k={}.".format(partition_k_pixel_id, nrows, subdivisions, iter_k))

    t1 = perf_counter()
    try:
        source_ids_to_cluster_id, cluster_id_to_data = find_clusters_one_partition(partition_k_pixel_id, neowise_ds, iter_k=iter_k)
    except Exception as e:
        log(f"Error on partition {partition_k_pixel_id}: {e}")
        print(f"Error on partition {partition_k_pixel_id}: {e}", file=open("/home/mpaz/neowise-clustering/clustering/logs/errors.txt", "a"))
        print(f"Error on partition {partition_k_pixel_id}: {e}", file=open("/home/mpaz/neowise-clustering/clustering/logs/masterlog.txt", "a"))
        continue
    t_clustering = perf_counter() - t1

    # Rename the cluster ID so instead of encoding (partition_id, position) it encodes (partition_id, order)
    # This is done so the cluster ID can be iterated upon

    n_apparitions_after_clustering = len(source_ids_to_cluster_id)
    n_clusters = cluster_id_to_data.num_rows
    log("Clustering on partition {} complete in {}s. {} clusters. {} apparitions included. It is {}".format(
        partition_k_pixel_id, t_clustering, n_clusters, n_apparitions_after_clustering, datetime.now()
        ))
    print(f"Partition {partition_k_pixel_id} complete in {t_clustering}s. {n_clusters} clusters. {n_apparitions_after_clustering} apparitions.",
    file=open("/home/mpaz/neowise-clustering/clustering/logs/masterlog.txt", "a"))

    open("/home/mpaz/neowise-clustering/clustering/logs/progress.txt", "a").write(f"{partition_k_pixel_id}_2_{nrows}_{n_apparitions_after_clustering}_{str(datetime.now())}\n")
    # Save the cluster map table to a file.

    PATH_TO_OUTPUT_DIRECTORY = "/home/mpaz/neowise-clustering/clustering/out"
    source_ids_to_cluster_id.to_csv(f"{PATH_TO_OUTPUT_DIRECTORY}/partition_{partition_k_pixel_id}_source_id_to_cluster_id.csv", index=False)
    pq.write_table(cluster_id_to_data, f"{PATH_TO_OUTPUT_DIRECTORY}/partition_{partition_k_pixel_id}_cluster_id_to_data.parquet")
