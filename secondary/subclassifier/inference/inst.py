import pandas as pd
import sys
sys.path.append("../")
from dataloader import TreeDL
import xgboost as xgb
from autoencoder import Morphologic
import torch
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow as pa
import os
from math import ceil
from joblib import Parallel, delayed
import time
import numpy as np

# seed
rng = np.random.default_rng(42)


partitions_to_process = list((set(range(12288)) - set([8448, 9728, 4010, 2559, 4460, 9386, 2901, 8277, 3839])) - set([int(s.split("_")[1][:-4]) for s in os.listdir("out") if s.endswith(".csv")]))
rng.shuffle(partitions_to_process)

SLURM_ARRAY_TASK_N = int(os.getenv("SLURM_ARRAY_TASK_ID"))
SLURM_ARRAY_TASK_COUNT = int(os.getenv("SLURM_ARRAY_TASK_COUNT"))
TOTAL_TASKS = len(partitions_to_process)
CHUNK_SIZE = ceil(TOTAL_TASKS / SLURM_ARRAY_TASK_COUNT)
partitions_to_process = partitions_to_process[SLURM_ARRAY_TASK_N * CHUNK_SIZE: (SLURM_ARRAY_TASK_N + 1) * CHUNK_SIZE]

BINS = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

model = Morphologic(BINS, 7, features=2) 
model.load_state_dict(torch.load("/home/mpaz/neovar/secondary/subclassifier/model/morpho.pth", map_location=device))

# load xgboost model
tree = xgb.Booster()
tree.set_param('nthread', 1)
tree.load_model("/home/mpaz/neovar/secondary/subclassifier/model/oldone.tree")
print('pre load complete')
print("processing ", len(partitions_to_process), " partitions")

def evaluate(partition_id):
    print("Processing partition ", partition_id)
    flag_data = pd.read_csv(f"/home/mpaz/neovar/inference/flagtbls/partition_{partition_id}_flag_tbl.csv")
    flag_data = flag_data[flag_data["extragalactic"] == False]
    flag_data = flag_data[flag_data["type"] == 2]
    print(len(flag_data))
    cids = flag_data["cluster_id"]
    flag_data.set_index("cluster_id", inplace=True)
    lc_data = pq.read_table(f"/home/mpaz/neowise-clustering/clustering/out/partition_{partition_id}_cluster_id_to_data.parquet",
                        filters=pc.is_in(pc.field("cluster_id"), pa.Array.from_pandas(cids)), use_threads=False).to_pandas().set_index("cluster_id")
    print('loaded {} rows'.format(len(lc_data)))

    data = pd.concat([lc_data, flag_data], axis=1)
    if len(data) != len(flag_data) or len(data) != len(lc_data):
        raise ValueError("Data length mismatch, likely indexing error")
    
    if len(data) == 0:
        print("No objects to process for partition ", partition_id)
        return

    tdl = TreeDL(data, model, training=False, lc_bins=BINS)

    feature_tbl = tdl.get_feature_tbl()
    tree_input = feature_tbl.to_numpy()
    result = tree.inplace_predict(tree_input, predict_type="margin")
    result = np.exp(result) / np.sum(np.exp(result), axis=1)[:, np.newaxis] # softmax for probabilities
    predictions = np.argmax(result, axis=1)
    score = np.max(result, axis=1)

    feature_tbl["tree_prediction"] = predictions
    feature_tbl["tree_score"] = score

    tps = tdl.types
    for i, tp in enumerate(tps):
        feature_tbl[f"{tp}_pred"] = result[:, i]

    feature_tbl.sort_values("tree_score", ascending=False, inplace=True)
    objs = feature_tbl.groupby("cluster_id")
    final = objs.first()
    final.to_csv(f"/home/mpaz/neovar/secondary/subclassifier/inference/out/partition_{partition_id}.csv")

    return final

results = Parallel(n_jobs=24)(delayed(evaluate)(partition_id) for partition_id in partitions_to_process)
# for p in partitions_to_process:
#     evaluate(p)