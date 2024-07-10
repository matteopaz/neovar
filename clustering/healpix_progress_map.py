import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import healpy as hp
import numpy as np
import time
import tqdm
import os

def ids_to_map():
    directory = os.fsencode("/home/mpaz/neowise-clustering/clustering/out/")
    partition_ids = [-1] * 12288
        
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"):
            parts = filename.split("_")
            pid = int(parts[1])
            partition_ids[pid] = 1

    return np.array(partition_ids)


def count_done(map):
    """Count the number of partitions that have been processed."""
    return len([x for x in map if x > 0])

def live_plot(interval=1):
    """Plot IDs from the given file, updating the plot every 'interval' seconds."""
    bar = tqdm.tqdm(total=12288)
    cmap = ListedColormap(['white', 'red', 'green'])
    # -1 = white = not processed
    # 0 = red = error
    # 1 = gree = completed

    while True:
        hpmap = ids_to_map()
        hp.visufunc.mollview(hpmap, title="Apparitions per partition", cmap=cmap, min=0, xsize=2000)
        hp.visufunc.graticule()
        # write current plot to a file
        plt.savefig("./logs/apparitions.png")
        
        bar.n = count_done(hpmap)
        bar.refresh()
        time.sleep(interval)


live_plot()