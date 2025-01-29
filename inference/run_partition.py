from periodic_analysis import classify_periodic
from transient_analysis import classify_transient
from load_data import PartitionDataLoader
import torch
import pandas as pd
import pickle
from model import WCNFourierModel
from time import perf_counter as pc
import warnings

torch.set_default_device("cuda")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
modelname = "newvalid_best"
params = pickle.load(open(f"/home/mpaz/neovar/inference/model/{modelname}.pkl", "rb"))
model = WCNFourierModel(**params).cuda()
model.load_state_dict(torch.load(f"/home/mpaz/neovar/inference/model/{modelname}.pt"))
model.eval().cuda()
BATCHSIZE = 384
CONFIDENCE_THRESH = 0.90

print("Model loaded: {}".format(modelname))

SAVE_DATA = True


def run_partition(partition_id: int):
    partition = pd.read_parquet(f"/home/mpaz/neowise-clustering/clustering/out/partition_{partition_id}_cluster_id_to_data.parquet")
    dataloader = PartitionDataLoader(partition, BATCHSIZE)
    subcatalogs = []
    datacatalogs = []

    i = 1
    for table, model_tensor in dataloader:
        print("Chunk {} out of {}".format(i, len(dataloader)))
        i += 1
        probs = model(model_tensor.cuda())
        probs = torch.softmax(probs, dim=1)
        maxes, l1_classes = torch.max(probs, dim=1)
        
        table["type"] = l1_classes.detach().cpu().numpy()
        table["type"] = table["type"].map({0: pd.NA, 1: "transient", 2: "periodic"})
        table["confidence"] = maxes.detach().cpu().numpy()

        table = table[table["type"].notna() & (table["confidence"] > CONFIDENCE_THRESH)]

        transient_subcatalog = classify_transient(table)
        periodic_subcatalog = classify_periodic(table)

        table = table[table["type"].notna()]

        if len(transient_subcatalog) > 0:
            subcatalogs.append(transient_subcatalog)
        if len(periodic_subcatalog) > 0:
            subcatalogs.append(periodic_subcatalog)
        if len(table) > 0:
            datacatalogs.append(table)
    
    partition_catalog = pd.concat(subcatalogs)
    partition_catalog.sort_values("confidence", ascending=False, inplace=True)

    datacatalog = pd.concat(datacatalogs)
    
    datacatalog.sort_values("confidence", ascending=False, inplace=True)

    return partition_catalog, datacatalog