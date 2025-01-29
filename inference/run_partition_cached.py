import torch
import pandas as pd
import numpy as np
import pickle
import json
from model import WCNFourierModel
import warnings
from math import ceil

torch.set_default_device("cuda")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
modelname = "newvalid_best"
params = pickle.load(open(f"/home/mpaz/neovar/inference/model/{modelname}.pkl", "rb"))
model = WCNFourierModel(**params).cuda()
model.load_state_dict(torch.load(f"/home/mpaz/neovar/inference/model/{modelname}.pt"))
model.eval().cuda()
BATCHSIZE = 384

# print("Model loaded: {}".format(modelname))

SAVE_DATA = True

def trim_tensor(tensor):
    toprow = tensor[0]
    firstzero = next((i for i, t in enumerate(toprow) if torch.sum(t) == 0), len(toprow))
    return tensor[:, :max(20, firstzero), :]

def run_partition(partition_id: int):
    # print("Loading data")
    try:
        tensor = torch.load(f"/home/mpaz/neowise-clustering/clustering/inference_ready/p{partition_id}_tensor.pt")
        cluster_ids = np.array(json.load(open(f"/home/mpaz/neowise-clustering/clustering/inference_ready/p{partition_id}_cluster_ids.json")))
    except:
        return
    # print("Data loaded")

    if len(tensor) != len(cluster_ids):
        raise ValueError("Mismatch between cluster_ids and tensor")

    n_sources = len(tensor)
    n_chunks = ceil(n_sources / BATCHSIZE)

    flag_tbls = []

    i = 1
    for i in range(n_chunks):
        # print("Chunk {} out of {}".format(i, n_chunks))
        start = i * BATCHSIZE
        end = (i + 1) * BATCHSIZE
        if n_sources % BATCHSIZE == 1:
            if i == n_chunks - 2:
                end += 1
            elif i == n_chunks - 1:
                continue
        data = trim_tensor(tensor[start:end])
        probs = model(data.cuda())
        if len(probs.shape) == 1:
            print(probs, partition_id, n_sources)
        probs = torch.softmax(probs, dim=1)
        maxes, classes = torch.max(probs, dim=1)
        
        table = pd.DataFrame({
            "cluster_id": cluster_ids[start:end],
            "type": classes.detach().cpu().numpy(),
            "confidence": maxes.detach().cpu().numpy()
        })
        table = table[(table["type"] != 0) & (table["confidence"] > 0.90)]
        flag_tbls.append(table)

    flag_tbl = pd.concat(flag_tbls)

    return flag_tbl