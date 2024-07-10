import os
import numpy as np
import matplotlib.pyplot as plt
import json

hist = np.zeros(1000)
for i in range(12):
    bins = list(json.load(open(f"/home/mpaz/neowise-clustering/clustering/maghistogram_{i}.json", "r")))
    hist += np.array(bins)

bins = np.linspace(0, 18, 1000)
hist = np.log10(hist + 1)
plt.bar(bins, hist)
plt.xlabel("Source Magnitude")
plt.ylabel("Count - Logscale 10^x")
plt.title("Histogram of magnitudes")
plt.savefig("/home/mpaz/neowise-clustering/clustering/maghistogram.png")
# hist is already binned
