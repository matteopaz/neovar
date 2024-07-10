import pyarrow as pa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from joblib import Parallel, delayed
from time import perf_counter
import os

MINMAG = 0
MAXMAG = 18

N_BINS = 1000

SLURM_ARRAY_TASK_ID = os.getenv("SLURM_ARRAY_TASK_ID")

mags = json.load(open(f"/home/mpaz/neowise-clustering/clustering/magnitudes_{SLURM_ARRAY_TASK_ID}.json", "r"))
mags = np.array(list(mags))

hist, bin_edges = np.histogram(mags, bins=N_BINS, range=(MINMAG, MAXMAG))

json.dump(hist.tolist(), open(f"/home/mpaz/neowise-clustering/clustering/maghistogram_{SLURM_ARRAY_TASK_ID}.json", "w"))

