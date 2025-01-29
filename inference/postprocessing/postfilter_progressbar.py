import pandas as pd
import os
import time
from joblib import Parallel, delayed

n_done = 12280

start_time = 1726079998

files = [f for f in os.listdir("./map_out/") if f.endswith(".csv")]
def count(file):
    with open(f"./map_out/{file}", "r") as f:
        line = f.readline()
        if len(line.split(",")) > 3:
            return 1
        else:
            return 0
    

n_total = sum(Parallel(n_jobs=12)(delayed(count)(file) for file in files))

# UNIX timestamp
now = time.time()
dt = now - start_time
rate = n_done / dt
left = n_total - n_done
eta = left / rate

print(f"Done: {n_done}/{n_total}")
print(f"Rate: {rate} partitions/s")
print(f"ETA: {eta} s")