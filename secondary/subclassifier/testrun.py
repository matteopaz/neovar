import pandas as pd
from dataloader import TreeDL
import xgboost as xgb
from autoencoder import Morphologic
import torch
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow as pa
from time import perf_counter
import time

BINS = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

model = Morphologic(BINS, 8, features=2)
model.load_state_dict(torch.load("/home/mpaz/neovar/secondary/subclassifier/model/morpho.pth", map_location=device))
# load xgboost model
tree = xgb.Booster()
tree.set_param('nthread', 1)
tree.load_model("/home/mpaz/neovar/secondary/subclassifier/model/10oclock.tree")
print('pre load complete')

def evaluate(partition_id):
    try:
        flag_data = pd.read_csv(f"/home/mpaz/neovar/inference/flagtbls/partition_{partition_id}_flag_tbl.csv")
        flag_data = flag_data[flag_data["extragalactic"] == False]
        flag_data = flag_data[flag_data["type"] == 2]
        flag_data = flag_data[flag_data["W1mag"] > 7]
        flag_data
        cids = flag_data["cluster_id"]
        flag_data.set_index("cluster_id", inplace=True)
        lc_data = pq.read_table(f"/home/mpaz/neowise-clustering/clustering/out/partition_{partition_id}_cluster_id_to_data.parquet",
                            filters=pc.is_in(pc.field("cluster_id"), pa.Array.from_pandas(cids)), use_threads=False).to_pandas().set_index("cluster_id")
        print('loaded {} rows'.format(len(lc_data)))
    except FileNotFoundError:
        return

    data = pd.concat([lc_data, flag_data], axis=1)
    if len(data) != len(flag_data) or len(data) != len(lc_data):
        raise ValueError("Data length mismatch, likely indexing error")
    
    tdl = TreeDL(data, model, training=False, lc_bins=BINS)

    feature_tbl = tdl.get_feature_tbl()
    tree_input = xgb.DMatrix(feature_tbl)
    predictions = tree.predict(tree_input)

    feature_tbl["tree_prediction"] = predictions

    feature_tbl.sort_values("period_significance", ascending=False, inplace=True)
    objs = feature_tbl.groupby("cluster_id")

    return objs.first()
