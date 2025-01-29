from autoencoder import *
from varnet import VARnet
import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd
import plotly.graph_objects as go
from dataloader import Trainer

# torch.multiprocessing.set_start_method('spawn')
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
torch.set_default_dtype(torch.float32)

training_data = pd.read_parquet('/home/mpaz/neovar/secondary/data/training_data.parquet')[::5]
print(len(training_data))
print(training_data.columns)

traind = []
validd = []
for type_ in training_data['type'].unique():
    indices = training_data['type'] == type_
    oftype = training_data[indices]
    traind.append(oftype.sample(frac=0.9))
    validd.append(oftype.drop(traind[-1].index))

traind = pd.concat(traind)
validd = pd.concat(validd)

print(training_data.value_counts('type'))
print(traind.value_counts('type'))
print(validd.value_counts('type'))

trainer = Trainer(traind, 128, 4096, False, True, multithread=True, bin_overlap_frac=0.5)
print(len(trainer))
valid = Trainer(validd, 128, 4096, False, True, multithread=True, bin_overlap_frac=0.5)
print(len(valid))

print(torch.sum(torch.isnan(trainer.tensor)))
print(torch.sum(torch.isinf(trainer.tensor)))

# pickle and save the trainer and valid
import pickle
with open('trainer.pkl', 'wb') as f:
    pickle.dump(trainer, f)
with open('valid.pkl', 'wb') as f:
    pickle.dump(valid, f)