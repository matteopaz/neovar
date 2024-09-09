import pandas as pd
import torch
import pickle
# add /home/mpaz/neovar/inference to the path
import sys
sys.path.append('/home/mpaz/neovar/inference')
from load_data import PartitionDataLoader

valid_data = pd.read_parquet('valid_data.parquet')

full_typemap = {
    "const": 0,
    "nova": 1,
    "sn": 2,
    "yso": 3,
    "agn": 4,
    "ceph": 5,
    "rvt": 5,
    "rr": 6,
    "mira": 7,
    "sr": 8,
    "ea": 9,
    "ew": 10
}

label_typemap ={
    "const": 0,
    "nova": 1,
    "sn": 1,
    "yso": 2,
    "agn": 2,
    "ceph": 2,
    "rvt": 2,
    "rr": 2,
    "mira": 2,
    "sr": 2,
    "ea": 2,
    "ew": 2,
    "mix": None
}


valid_data['label'] = valid_data['type'].apply(lambda x: label_typemap[x])
valid_data = valid_data.dropna(subset=['label'])
# valid_data['type'] = valid_data['type'].apply(lambda x: full_typemap[x])

pdl = PartitionDataLoader(valid_data, len(valid_data))

cleantbl, tensor = next(iter(pdl))
labels = cleantbl["label"].values
label = torch.zeros((len(labels),3))
label[list(range(len(labels))), labels] = 1 # target tensor


print("Longest sequence:", cleantbl.iloc[0]["npts"])
print("Shortest sequence:", cleantbl.iloc[-1]["npts"])

pickle.dump((tensor, label, cleantbl["type"].values, cleantbl["source_id"].values), open('valid_data.pkl', 'wb'))

