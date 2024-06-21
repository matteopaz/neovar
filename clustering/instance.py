import pickle
import argparse
from time import perf_counter
from func import find_clusters_one_partition
import pyarrow.parquet as pq

# parse arg --partition_id
parser = argparse.ArgumentParser()
parser.add_argument("--partition_id", type=int)
args = parser.parse_args()

partition_k_pixel_id = int(args.partition_id)

print("Beginning partition {}".format(partition_k_pixel_id))

neowise_ds = pickle.load(open("neowise_ds.pkl", "rb"))

start = perf_counter()
source_ids_to_cluster_id, cluster_id_to_source_ids = find_clusters_one_partition(partition_k_pixel_id, neowise_ds)
dt = perf_counter() - start

# Rename the cluster ID so instead of encoding (partition_id, position) it encodes (partition_id, order)
# This is done so the cluster ID can be iterated upon


n_apparitions_after_clustering = len(source_ids_to_cluster_id)
n_clusters = len(cluster_id_to_source_ids)
print("Clustering on partition {} complete in {}s. {} clusters. {} apparitions included".format(
    partition_k_pixel_id, dt, n_clusters, n_apparitions_after_clustering)
    )

# Save the cluster map table to a file.

PATH_TO_OUTPUT_DIRECTORY = "/home/mpaz/neowise-clustering/clustering/out"
source_ids_to_cluster_id.to_csv(f"{PATH_TO_OUTPUT_DIRECTORY}/partition_{partition_k_pixel_id}_source_id_to_cluster_id.csv", index=False)
pq.write_table(cluster_id_to_source_ids, f"{PATH_TO_OUTPUT_DIRECTORY}/partition_{partition_k_pixel_id}_cluster_id_to_source_id.parquet")
print("Results accessible at /home/mpaz/neowise-clustering/clustering/out/partition_{}_cluster_map.csv".format(partition_k_pixel_id))
