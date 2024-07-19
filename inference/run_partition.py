from periodic_analysis import classify_periodic
from transient_analysis import classify_transient
from load_data import PartitionDataLoader
import torch
import pandas as pd
import pickle
from model import CNFourierModel
from time import perf_counter as pc
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch")
modelname = "GOOD"
params = pickle.load(open(f"/local/home/mpaz/neovar/inference/model/{modelname}.pkl", "rb"))
model = CNFourierModel(**params).cuda()
model.load_state_dict(torch.load(f"/local/home/mpaz/neovar/inference/model/{modelname}.pt"))
model.eval().cuda()
BATCHSIZE = 2048

print("Model loaded: {}".format(modelname))


def run_partition(partition_id: int):
    partition = pd.read_parquet(f"./in/partition_{partition_id}_cluster_id_to_data.parquet")
    dataloader = PartitionDataLoader(partition, BATCHSIZE, prefilter=False)
    subcatalogs = []

    i = 1
    for table, model_tensor in dataloader:
        print("Chunk {} out of {}".format(i, len(dataloader)))
        i += 1
        probs = model(model_tensor.cuda())
        probs = torch.softmax(probs, dim=1)
        maxes, l1_classes = torch.max(probs, dim=1)
        confident = maxes > 0.8
        maxes = maxes.squeeze()
        table["conf"] = maxes.detach().cpu().numpy()
        l1_classes = l1_classes.squeeze()

        transient_indices = torch.logical_and(l1_classes == 1, confident).nonzero(as_tuple=True)[0].cpu()
        periodic_indices = torch.logical_and(l1_classes == 2, confident).nonzero(as_tuple=True)[0].cpu()

        transients = table.iloc[transient_indices].copy()
        periodic = table.iloc[periodic_indices].copy()

        transient_subcatalog = classify_transient(transients)
        periodic_subcatalog = classify_periodic(periodic)

        if len(transient_subcatalog) > 0:
            subcatalogs.append(transient_subcatalog)
        if len(periodic_subcatalog) > 0:
            subcatalogs.append(periodic_subcatalog)
    
    partition_catalog = pd.concat(subcatalogs)
    partition_catalog.sort_values("confidence", ascending=False, inplace=True)

    return partition_catalog