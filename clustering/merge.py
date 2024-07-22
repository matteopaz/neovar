import os 
import pandas as pd
from collections import defaultdict

dir = os.fsencode("/home/mpaz/neowise-clustering/clustering/out/")

merging_partitions = defaultdict(set)

for file in os.listdir(dir):
    filename = os.fsdecode(file)
    parts = filename.split("_")
    if len(parts) > 6:
        pid = int(parts[1])
        merging_partitions[pid].add(filename)

# for pid, files in merging_partitions.items():
#     if not os.path.isfile(f"/home/mpaz/neowise-clustering/clustering/out/partition_{pid}_cntr_to_cluster_id.csv"):
#         print("Partition {} did not complete".format(pid))
#     else:
#         pass
        # csvs = [pd.read_csv(f"/home/mpaz/neowise-clustering/clustering/out/{file}") for file in files if file.endswith(".csv")]
        # merged = pd.concat(csvs)
        # merged.to_csv(f"/home/mpaz/neowise-clustering/clustering/out/partition_{pid}_cntr_to_cluster_id.merged.csv", index=False)

        # parquet = [pd.read_parquet(f"/home/mpaz/neowise-clustering/clustering/out/{file}") for file in files if file.endswith(".parquet")]
        # merged = pd.concat(parquet)
        # merged.to_parquet(f"/home/mpaz/neowise-clustering/clustering/out/partition_{pid}_cluster_id_to_data.merged.parquet")
            
print(merging_partitions)