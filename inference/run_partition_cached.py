import torch
import pandas as pd
import numpy as np
import pickle
import json
from model import WCNFourierModel
import warnings

torch.set_default_device("cuda")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
modelname = "newvalid_best"
params = pickle.load(open(f"/home/mpaz/neovar/inference/model/{modelname}.pkl", "rb"))
model = WCNFourierModel(**params).cuda()
model.load_state_dict(torch.load(f"/home/mpaz/neovar/inference/model/{modelname}.pt"))
model.eval().cuda()
BATCHSIZE = 384

print("Model loaded: {}".format(modelname))

SAVE_DATA = True


def run_partition(partition_id: int):
    # print("Loading data")
    tensor = torch.load(f"/home/mpaz/neowise-clustering/clustering/inference_ready/p{partition_id}_tensor.pt")
    cluster_ids = np.array(json.load(open(f"/home/mpaz/neowise-clustering/clustering/inference_ready/p{partition_id}_cluster_ids.json")))
    # print("Data loaded")

    if len(tensor) != len(cluster_ids):
        raise ValueError("Mismatch between cluster_ids and tensor")

    n_sources = len(tensor)
    n_chunks = n_sources // BATCHSIZE + 1

    flag_tbls = []

    i = 1
    for i in range(n_chunks):
        # print("Chunk {} out of {}".format(i, n_chunks))
        data = tensor[i * BATCHSIZE:(i + 1) * BATCHSIZE] # 
        probs = model(data.cuda())
        probs = torch.softmax(probs, dim=1)
        maxes, classes = torch.max(probs, dim=1)
        
        table = pd.DataFrame({
            "cluster_id": cluster_ids[i * BATCHSIZE:(i + 1) * BATCHSIZE],
            "type": classes.detach().cpu().numpy(),
            "confidence": maxes.detach().cpu().numpy()
        })
        table = table[(table["type"] != 0) & (table["confidence"] > 0.9)]
        flag_tbls.append(table)

    flag_tbl = pd.concat(flag_tbls)

    return flag_tbl