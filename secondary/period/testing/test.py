import pandas as pd
import numpy as np
import hpgeom.hpgeom as hpg
import pyarrow as pa
import pyarrow.parquet as pq
import sys
sys.path.append("../")
from get_period_improved import get_period
import matplotlib.pyplot as plt
import astropy.coordinates as coord
import astropy.units as u
import time
from joblib import Parallel, delayed
import pickle

trialpds = np.concatenate([
    np.arange(0.1, 4, 0.00005),
    np.arange(4, 10, 0.00007),
    np.arange(10, 30, 0.0005),
    np.arange(30, 50, 0.001),
    np.arange(50, 100, 0.005),
    np.arange(100, 500, 0.025),
    np.arange(500, 1000, 0.5),
])

print(len(trialpds))

cat = pd.read_csv("periodic_variables.csv")
cat["designation"] = cat["designation"].apply(lambda x: x.strip())
# temp = cat.iloc[:2]
ofinterest = "VarWISE J053315.06-013327.6"

vwise = pd.read_csv("/home/mpaz/neovar/final/catalogs/VarWISE.csv")

vwise.set_index("Designation", inplace=True)

cat["cluster_id"] = vwise.loc[cat["designation"].values]["cluster_id"].values
cat["ra"] = vwise.loc[cat["designation"].values]["RAJ2000"].values
cat["dec"] = vwise.loc[cat["designation"].values]["DecJ2000"].values
cat["partition"] = hpg.angle_to_pixel(32, cat["ra"], cat["dec"])

cat.set_index("cluster_id", inplace=True)

def data_tbl_apply_mask(data_tbl):
    if "filter_mask" not in data_tbl.columns:
        raise KeyError("data_tbl must have a 'filter_mask' column")
    def applymask(row):
        mask = row["filter_mask"]
        if not isinstance(mask, np.ndarray):
            raise ValueError("Filter mask is invalid - possibly missing?")
        for key in row.keys():
            if key == "filter_mask":
                continue
            itm = row[key]
            if isinstance(itm, np.ndarray):
                row[key] = itm[mask]
        return row
    return data_tbl.apply(applymask, axis=1)

def upscale_period_grid(periods, trialperiods, n):
    # Find index of nearest value in trialperiods
    dmatrix = np.abs(np.subtract.outer(periods, trialperiods))
    closest_idx = np.argmin(dmatrix, axis=1)
    # Retrieve period values of neighboring indices and finely interpolate
    newgrid = []
    for c in closest_idx:
        idx_before = max(0, c - 1)
        idx_after = min(len(trialperiods) - 1, c + 1)
        lowbound = trialperiods[idx_before]
        highbound = trialperiods[idx_after]
        neighborhood = np.linspace(lowbound, highbound, n)
        newgrid.extend(neighborhood.tolist())
    newgrid = np.array(newgrid, dtype=np.float32)
    return np.array(list(set(newgrid.tolist())), dtype=np.float32) # Remove repeated values as a result of 32-bit float precision


for pid, group in cat.groupby("partition"):
    data = pq.read_table(f"/home/mpaz/neowise-clustering/clustering/out/partition_{pid}_cluster_id_to_data.parquet").to_pandas()
    data = data_tbl_apply_mask(data)
    data.set_index("cluster_id", inplace=True)
    t = data["mjd"]
    y = data["w1flux"].apply(lambda x: (x - np.mean(x)) / np.std(x))

    cat.loc[group.index, "time"] = t
    cat.loc[group.index, "mag"] = y

cat.to_parquet("OUT+DATA.parquet")

print("Periodfinding")
t1 = time.time()
unrefined_period_table, pgram = get_period(cat, trialpds, return_pgram=True, peak_resolution_pct=6)
t2 = time.time()
print("First-stage periodfinding took", t2 - t1, "seconds")
print(len(cat), " objects")
rate = (t2 - t1) / len(cat)
print("Rate: ", rate, " seconds per object")

t1 = time.time()
refine_pds = []
mask = []
print(unrefined_period_table)
for i, (_, row) in enumerate(unrefined_period_table.iterrows()):
    periods = [row["peak1"], row["peak2"], row["peak3"]]
    refined = upscale_period_grid(periods, trialpds, 100)
    refine_pds.extend(refined)
    mask.extend([i]*len(refined))

refine_pds = np.array(refine_pds, dtype=np.float32)
mask = np.array(mask, dtype=np.int32)
_, unique_idx = np.unique(refine_pds, return_index=True)
refine_pds = refine_pds[unique_idx]
mask = mask[unique_idx]
sorter = np.argsort(refine_pds)
refine_pds = refine_pds[sorter]
mask = mask[sorter]


refined_period_table, refined_periodogram = get_period(cat, refine_pds, return_pgram=True, peak_resolution_pct=5, mask=mask)
t2 = time.time()
print("Refinement took", t2 - t1, "seconds")
rate = (t2 - t1) / len(cat)
print("Rate: ", rate, " seconds per object")

period_table = {"peak1": [], "peak2": [], "peak3": [], "peak1_sig": [], "peak2_sig": [], "peak3_sig": [], "cluster_id": []}

for idx in unrefined_period_table.index: # Combines refined periods with original periods, making sure not to overwrite unrefined long-periods
    if cat.loc[idx]["designation"] == ofinterest:
        print("Found")
        print(unrefined_period_table.loc[idx])
        print(refined_period_table.loc[idx])
        
        
    row_unrefined = unrefined_period_table.loc[idx]
    row_refined = refined_period_table.loc[idx]
    
    pds = np.array([row_refined["peak1"], row_refined["peak2"], row_refined["peak3"]] + [row_unrefined["peak"+str(i)] for i in range(1,4) if row_unrefined["peak"+str(i)] > 50])
    sigs = np.array([row_refined["peak1_sig"], row_refined["peak2_sig"], row_refined["peak3_sig"]] + [row_unrefined["peak"+str(i)+"_sig"] for i in range(1,4) if row_unrefined["peak"+str(i)] > 50])
    best_3 = np.argsort(sigs)[::-1][:3]

    best_pds = pds[best_3]
    best_sigs = sigs[best_3]
    period_table["peak1"].append(best_pds[0])
    period_table["peak2"].append(best_pds[1])
    period_table["peak3"].append(best_pds[2])
    period_table["peak1_sig"].append(best_sigs[0])
    period_table["peak2_sig"].append(best_sigs[1])
    period_table["peak3_sig"].append(best_sigs[2])

    period_table["best_long_period"] = row_unrefined["best_long_period"]
    period_table["best_long_period_sig"] = row_unrefined["best_long_period_sig"]
    period_table["cluster_id"].append(idx)

period_table = pd.DataFrame(period_table)
period_table.set_index("cluster_id", inplace=True)

cat = cat.join(period_table)
period_table.to_csv("period_table.csv")

cat["period"] = cat['period'].astype(float)

with open("out/results.pkl", "wb") as f:
    pickle.dump((cat, pgram), f)

def fold_for_axis(t, y, p):
    x = t % p
    x = x / p
    x = np.concatenate([x, x + 1])
    y = np.concatenate([y, y])
    return x,y

for i, row in cat.reset_index().iterrows():

    title = row["designation"] + "ref"
    t = row["time"]
    y = row["mag"]
    p_t = row["period"]

    p1, ps1 = row["peak1"], row["peak1_sig"]
    p2, ps2 = row["peak2"], row["peak2_sig"]
    plp, pslp = row["best_long_period"], row["best_long_period_sig"]

    # make 4-part subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # top right: full light curve
    axs[0, 0].plot(t, y, ".", label="Full Light Curve")
    axs[0, 1].plot(*fold_for_axis(t, y, p_t), ".", label="Correct Period: {}".format(p_t))
    axs[1, 0].plot(*fold_for_axis(t, y, p1), ".", label=f"Peak 1: {p1} @ sig {ps1}")
    axs[1, 1].plot(*fold_for_axis(t, y, p2), ".", label=f"Peak 2: {p2} @ sig {ps2}")

    # title each subplot
    axs[0, 0].set_title("Full Light Curve")
    axs[0, 1].set_title("Correct Period: %.5f" % p_t)
    axs[1, 0].set_title(f"Peak 1: {p1} @ sig {np.round(ps1, 3)}")
    axs[1, 1].set_title(f"Peak 2: {p2} @ sig {np.round(ps2, 3)}")

    fig.savefig("out/{}.png".format(title))

    trialpds = pgram[0]
    sigls = pgram[1][0]
    sigls = np.concatenate((np.zeros((sigls.shape[0],len(trialpds) - sigls.shape[1])), sigls), axis=1)
    sigce = pgram[1][1]
    sigaov = pgram[1][2]
    sigaov = np.concatenate((sigaov, np.zeros((sigaov.shape[0], len(trialpds) - sigaov.shape[1]))), axis=1)

    sigce = np.zeros_like(sigaov)



    density = 1 / (np.diff(trialpds))
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0,0].plot(np.log10(trialpds[:-1]), density)
    axs[0,0].set_title("Density")
    x = np.log10(trialpds)    

    truepd = row["period"]
    true = [np.log10(truepd)]*2

    # plot only within 10% of the true period
    filt = np.where(np.abs(trialpds - truepd) < 0.1 * truepd)[0]
    x = x[filt]
    sigls = sigls[:, filt]
    sigce = sigce[:, filt]
    sigaov = sigaov[:, filt]


    axs[0,1].plot(x, sigls[i])
    axs[0,1].plot(true, (np.min(sigls[i]), np.max(sigls[i])), label="True Period", linestyle=":")
    axs[0,1].set_title("Lomb-Scargle")
    axs[1,0].plot(x, sigce[i])
    axs[1,0].plot(true, (np.min(sigce[i]), np.max(sigce[i])), label="True Period", linestyle=":")
    axs[1,0].set_title("Conditional Entropy")
    axs[1,1].plot(x, sigaov[i])
    axs[1,1].plot(true, (np.min(sigaov[i]), np.max(sigaov[i])), label="True Period", linestyle=":")
    axs[1,1].set_title("Analysis of Variance")

    fig.savefig("out/{}_pgram.png".format(title))
    plt.close("all")

    