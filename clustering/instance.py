import pickle
import argparse
from func import find_clusters_one_partition

# parse arg --partition_id
parser = argparse.ArgumentParser()
parser.add_argument("--partition_id", type=int)
args = parser.parse_args()

partition_k_pixel_id = int(args.partition_id)

neowise_ds = pickle.load(open("neowise_ds.pkl", "rb"))

cluster_map_tbl = find_clusters_one_partition(partition_k_pixel_id, neowise_ds)

# Save the cluster map table to a file.

PATH_TO_OUTPUT_DIRECTORY = "/home/mpaz/neowise-clustering/out"
cluster_map_tbl.to_csv(f"{PATH_TO_OUTPUT_DIRECTORY}/partition_{partition_k_pixel_id}_cluster_map.csv", index=False)