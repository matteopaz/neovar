import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import healpy as hp
import numpy as np
import time
import tqdm

def ids_to_map(file_path):
    """Read IDs from the given file and return them as a list of integers."""
    hpmap = [0] * (12288)
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.split("_")
            if len(parts) < 4:
                continue
            id = parts[0]
            status = int(parts[1]) + 1
            hpmap[int(id)] = (status)
    return np.array(hpmap)

def count_done(map):
    """Count the number of partitions that have been processed."""
    return len([x for x in map if x > 0])

def live_plot(file_path, interval=1):
    """Plot IDs from the given file, updating the plot every 'interval' seconds."""
    bar = tqdm.tqdm(total=12288)
    cmap = ListedColormap(['white', 'red', 'green'])
    # -1 = white = not processed
    # 0 = red = error
    # 1 = gree = completed

    while True:
        hpmap = ids_to_map(file_path)
        hp.visufunc.mollview(hpmap, title="Apparitions per partition", cmap=cmap, min=0, xsize=2000)
        hp.visufunc.graticule()
        # write current plot to a file
        plt.savefig("./logs/apparitions.png")
        
        bar.n = count_done(hpmap)
        bar.refresh()
        time.sleep(interval)


live_plot("./logs/progress.txt")