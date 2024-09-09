import pandas as pd
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt

# Galactic Center
# partitiontbl = pd.read_parquet("/home/mpaz/neowise-clustering/clustering/out/partition_7207_cluster_id_to_data.parquet")
# Orion
# partitiontbl = pd.read_parquet("/home/mpaz/neowise-clustering/clustering/out/partition_5359_cluster_id_to_data.parquet")
# Bootes
# partitiontbl = pd.read_parquet("/home/mpaz/neowise-clustering/clustering/out/partition_2186_cluster_id_to_data.parquet")
# partitiontbl = pd.read_parquet("/home/mpaz/neowise-clustering/clustering/out/partition_0_cluster_id_to_data.parquet")

# partitions = [0, 5359, 2186, 10, 12, 11512, 9090, 4354, 29, 18]
partitions = [10240, 10242, 10248, 10250, 10272, 10280, 10282, 10368, 10370,
       10378, 10400, 10402, 10408, 10752, 10754, 10760, 10762, 10784,
       10792, 10794, 10880, 10882, 10888, 10890, 10914, 10920, 10922,
        6144,  6144,  6147,  6156,  6156,  6159,  6170,  6192,  6195,
        6204,  6204,  6207,  6336,  6336,  6339,  6348,  6351,  6351,
        6384,  6387,  6396,  6396,  6399,  6912,  6915,  6915,  6924,
        6927,  6960,  6960,  6963,  6972,  6975,  6975,  7104,  7107,
        7107,  7116,  7119,  7130,  7152,  7155,  7155,  7164,  7167,
        7167,  2730,  2731,  2734,  2746,  2747,  2750,  2751,  2794,
        2795,  2799,  2810,  2811,  2814,  2815,  2987,  2990,  2991,
        3002,  3006,  3007,  3050,  3051,  3055,  3066,  3067,  3070,
        3071]

partitiontbl = pd.concat([pd.read_parquet(f"/home/mpaz/neowise-clustering/clustering/out/partition_{p}_cluster_id_to_data.parquet") for p in partitions])

def getshortcadence(timestamps):
    dt = np.diff(timestamps)
    short = dt[dt < 10]
    return np.mean(short)

def getlongcadence(timestamps):
    dt = np.diff(timestamps)
    long = dt[dt >= 10]
    return np.mean(long)

short = partitiontbl["mjd"].apply(getshortcadence).values
long = partitiontbl["mjd"].apply(getlongcadence).values
timespan = partitiontbl["mjd"].apply(lambda x: x[-1] - x[0]).values

plt.plot(short, long, "b.")
plt.savefig("shortvlong.png")

# plot histograms of all three
# plt.hist(short, bins=100, alpha=0.5, label="Short Cadence", density=True)
# plt.savefig("shortcadence.png")
# plt.close()
# plt.hist(long, bins=100, alpha=0.5, label="Long Cadence", density=True)
# plt.savefig("longcadence.png")
# plt.close()
# plt.hist(timespan, bins=100, alpha=0.5, label="Timespan", density=True)
# plt.savefig("timespan.png")
# plt.close()

# print statistics
# print(f"Short Cadence Median: {np.median(short)}")
# print(f"Long Cadence Median: {np.median(long)}")
# print(f"Timespan Median: {np.median(timespan)}")
