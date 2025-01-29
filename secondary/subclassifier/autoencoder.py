import torch
import torch.nn as nn
import sys
sys.path.append("/home/mpaz/neovar/inference/")
from model import WCNFourierModel, FDFT
from load_data import PartitionDataLoader
import pandas as pd
from torch.utils.data import IterableDataset
import numpy as np
import json
from dataloader import data_tbl_to_nn_input

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float32)
default_dtype = torch.get_default_dtype()


class AutoEncoder(nn.Module):
    def __init__(self, bins: int, latent: int, features: int = 2):
        super().__init__()
        self.bins = bins
        self.latent = latent

        self.features = features    
        features = self.features

        self.enc_conv_1 = nn.Conv1d(in_channels=features, out_channels=features, kernel_size=5, padding="same", padding_mode="circular")
        self.enc_conv_2 = nn.Conv1d(in_channels=features, out_channels=2, kernel_size=5, padding="same", padding_mode="circular")
        self.enc_lin_1 = nn.Linear(2*bins, (bins // 2))
        self.enc_lin_2 = nn.Linear((bins // 2), latent)

        self.dec_lin_1 = nn.Linear(latent, (bins // 2))
        self.dec_lin_2 = nn.Linear((bins // 2), 2*bins)
        self.dec_conv_1 = nn.Conv1d(in_channels=2, out_channels=features, kernel_size=5, padding="same", padding_mode="circular")
        self.dec_conv_2 = nn.Conv1d(in_channels=features, out_channels=features, kernel_size=3, padding="same", padding_mode="circular")

        self.av = lambda x: nn.functional.leaky_relu(x, 0.2)

    def encode(self, x):
        if x.shape[1] != self.bins:
            raise ValueError(f"Expected {self.bins} bins, got {x.shape[1]}")
        elif x.shape[2] != self.features:
            raise ValueError(f"Expected {self.features} features, got {x.shape[2]}")
        
        conv_x = x.permute(0, 2, 1)
        c1 = self.av(self.enc_conv_1(conv_x))
        c2 = self.av(self.enc_conv_2(c1))
        flat = torch.cat([c2[:, 0, :], c2[:, 1, :]], dim=1)
        l1 = self.av(self.enc_lin_1(flat))
        l2 = self.enc_lin_2(l1)
        return l2

    def decode(self, latent):
        l1 = self.av(self.dec_lin_1(latent))
        l2 = self.av(self.dec_lin_2(l1))
        channels = torch.stack([l2[:, :self.bins], l2[:, self.bins:]], dim=1)
        c1 = self.av(self.dec_conv_1(channels))
        final = self.dec_conv_2(c1)
        return final.permute(0, 2, 1)

    def forward(self, x):
        latent = self.encode(x)
        original = self.decode(latent)
        return original


class Morphologic(nn.Module):
    def __init__(self, bins: int, out: int, features: int = 2):
        super().__init__()
        self.bins = bins
        self.out = out

        self.features = features    
        features = self.features

        self.c1 = nn.Conv1d(in_channels=features+2, out_channels=8, kernel_size=5, padding="same", padding_mode="circular")
        self.c2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=5, padding="same", padding_mode="circular")
        self.c3 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=5, padding="same", padding_mode="circular")
        self.c4 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=5, padding="same", padding_mode="circular")
        self.l1 = nn.Linear(8*bins, 4*bins)
        self.l2 = nn.Linear(4*bins, 2*bins)
        self.l3 = nn.Linear(2*bins, bins)
        self.l4 = nn.Linear(bins, bins//2)
        self.l5 = nn.Linear(bins//2, out)
        self.av = nn.ReLU()

    def forward(self, x):
        if x.shape[1] != self.bins:
            raise ValueError(f"Expected {self.bins} bins, got {x.shape[1]}")
        elif x.shape[2] != self.features:
            raise ValueError(f"Expected {self.features} features, got {x.shape[2]}")
        
        fft_complex = torch.fft.fft(x[:,:,0])
        fft = torch.stack([fft_complex.real, fft_complex.imag], dim=2)
        x = torch.cat((x, fft), dim=2)

        conv_x = x.permute(0, 2, 1)
        c1 = self.av(self.c1(conv_x))
        c2 = self.av(self.c2(c1))
        c3 = self.av(self.c3(c2))
        c4 = self.av(self.c4(c3))
        flat = torch.cat([c4[:, i, :] for i in range(c4.shape[1])], dim=1)

        l1 = self.av(self.l1(flat))
        l2 = self.av(self.l2(l1))
        l3 = self.av(self.l3(l2))
        l4 = self.av(self.l4(l3))
        l5 = self.l5(l4)

        return l5