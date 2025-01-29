import pandas as pd
import os
from joblib import Parallel, delayed
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import astropy

def process(pid):
    try:
        df = pd.read_csv(f"/home/mpaz/neovar/inference/out2_filtered/partition_{pid}_flag_tbl.csv")
    except:
        return None
    
    og_len = len(df)
    df = df[df['type'] == 1]
    # df = df[df["extragalactic"] == True]
    # if 'latent_flag' not in df.columns:
    #     if len(df) != 0:
    #         print("No latent flag in", pid)
    #     return None
    # df = df[df["latent_flag"] != True]

    if "ra" not in df.columns:
        return None

    ra = df['ra'].values
    dec = df['dec'].values

    return len(df), ra, dec


prods = Parallel(n_jobs=-2)(delayed(process)(i) for i in tqdm.tqdm(range(12288)))

count = np.array([p[0] for p in prods if p is not None])
ra = np.concatenate([p[1] for p in prods if p is not None])
dec = np.concatenate([p[2] for p in prods if p is not None])

top = np.argsort(count)[-10:].astype(int)
# print(top)
# print(count[top])

print(np.sum(count))
fig = plt.figure()
plt.subplot(111, projection="mollweide")
plt.scatter(x=np.radians(ra), y=np.radians(dec), s=0.05, alpha=0.5)
plt.grid(False)
# for each pid in problems, plot a red dot

# plot in 4k resolution
fig.set_size_inches(20, 10)
fig.savefig("allsky_transient.png")