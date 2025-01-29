import torch
import pandas as pd
import numpy as np
import json
from load_data import PartitionDataLoader

partition_id = 0

tensor = torch.load(f"/home/mpaz/neowise-clustering/clustering/inference_ready/p{partition_id}_tensor.pt").cuda()
cluster_ids = np.array(json.load(open(f"/home/mpaz/neowise-clustering/clustering/inference_ready/p{partition_id}_cluster_ids.json")))

partition = pd.read_parquet(f"/home/mpaz/neowise-clustering/clustering/out/partition_{partition_id}_cluster_id_to_data.parquet")
dataloader = PartitionDataLoader(partition, 1)

parts = [p for p in dataloader]
tbl = pd.concat([p[0] for p in parts])
tensors = [p[1].squeeze() for p in parts]
model_tensor = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
model_tensor = model_tensor.cuda()
cids = tbl["cluster_id"].values

print("CIDS set-same: ", set(cids) == set(cluster_ids))
print("Tensor same:", torch.sum(tensor - model_tensor).item() == 0)

