import pandas as pd
import torch
import pickle
from neovar.inference.load_data import PartitionDataLoader

valid_data = pd.read_parquet('valid_data.parquet')

full_typemap = {
    "null": 0,
    "nova": 1,
    "sn": 2,
    "yso": 3,
    "agn": 4,
    "cep": 5,
    "rvt": 5,
    "rr": 6,
    "mira": 7,
    "sr": 8,
    "ea": 9,
    "ew": 10
}

label_typemap ={
    "null": 0,
    "nova": 1,
    "sn": 1,
    "yso": 3,
    "agn": 2,
    "cep": 2,
    "rr": 2,
    "mira": 2,
    "sr": 2,
    "ea": 2,
    "ew": 2
}

n_classes = len(set(full_typemap.values()))

valid_data['label'] = valid_data['type'].apply(lambda x: label_typemap[x])
valid_data['type'] = valid_data['type'].apply(lambda x: full_typemap[x])

pdl = PartitionDataLoader(valid_data, len(valid_data), prefilter=True)

cleantbl, tensor = next(iter(pdl))
labels = cleantbl["label"].values
label = torch.zeros((len(labels),n_classes))
label[list(range(len(labels))), labels] = 1 # target tensor



print("Longest sequence:", cleantbl.iloc[0]["nrows"])
print("Shortest sequence:", cleantbl.iloc[-1]["nrows"])

pickle.dump((tensor, label, cleantbl["type"].values), open('valid_data.pkl', 'wb'))

