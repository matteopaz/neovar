from datetime import datetime
from time import perf_counter, sleep
import os
import argparse
from math import ceil, log2
from random import randint

MAX_ROWS_TO_LOAD = 8 * 10**6 # 7 Million

def sizeof(partition_id):
    return 499 * 10**6

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

def log(*args):
    with open(PATH_TO_OUTFILE, "a") as f:
        print(*args, file=f)

end_idx = min(end_idx, 12287)

log("Slurm job {} with task ID {} assigned on partitions {} to {}".format(SLURM_JOB_ID, SLURM_ARRAY_TASK_N, start_idx, end_idx))

for partition_k_pixel_id in range(start_idx, end_idx):
    log("Beginning clustering on partition {} at {}".format(partition_k_pixel_id, datetime.now()))
    open("/home/mpaz/neowise-clustering/clustering/logs/progress.txt", "a").write(f"{partition_k_pixel_id}_1_{str(datetime.now())}\n")

    nrows = sizeof(partition_k_pixel_id)
    subdivisions = ceil(0.5 * log2(nrows / MAX_ROWS_TO_LOAD)) # Log 4 of the number of rows
    iter_k = 5 + subdivisions
    log("Partition {} has {} rows. Using {} subdivisions - iter_k={}.".format(partition_k_pixel_id, nrows, subdivisions, iter_k))

    t1 = perf_counter()
    sleep(0.1)
    try:
        if randint(0, 25) == 0:
            raise Exception("Random exception")
    except Exception as e:
        open("/home/mpaz/neowise-clustering/clustering/logs/progress.txt", "a").write(f"{partition_k_pixel_id}_0_{str(datetime.now())}\n")
        log(f"Error on partition {partition_k_pixel_id}: {e}")
        print(f"Error on partition {partition_k_pixel_id}: {e}", file=open("/home/mpaz/neowise-clustering/clustering/logs/errors.txt", "a"))

        continue

    t_clustering = perf_counter() - t1

    # Rename the cluster ID so instead of encoding (partition_id, position) it encodes (partition_id, order)
    # This is done so the cluster ID can be iterated upon

    n_apparitions_after_clustering = 999
    n_clusters = 999
    log("Clustering on partition {} complete in {}s. {} clusters. {} apparitions included.".format(
        partition_k_pixel_id, t_clustering, n_clusters, n_apparitions_after_clustering)
        )
    
    print(f"Partition {partition_k_pixel_id} complete in {t_clustering}s. {n_clusters} clusters. {n_apparitions_after_clustering} apparitions included.",
    file=open("/home/mpaz/neowise-clustering/clustering/logs/masterlog.txt", "a"))

    open("/home/mpaz/neowise-clustering/clustering/logs/progress.txt", "a").write(f"{partition_k_pixel_id}_2_{str(datetime.now())}\n")
    # Save the cluster map table to a file.

    PATH_TO_OUTPUT_DIRECTORY = "/home/mpaz/neowise-clustering/clustering/out"
    # source_ids_to_cluster_id.to_csv(f"{PATH_TO_OUTPUT_DIRECTORY}/partition_{partition_k_pixel_id}_source_id_to_cluster_id.csv", index=False)
    # pq.write_table(cluster_id_to_data, f"{PATH_TO_OUTPUT_DIRECTORY}/partition_{partition_k_pixel_id}_cluster_id_to_data.parquet")
