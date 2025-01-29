#
# Columns wanted: Designation, RA, Dec, W1mag, W2mag, amplitude, variability_snr, period1, period2, period3, likelihood, possibly_blended, type
#
#

import pandas as pd
import numpy as np
import os
from astropy.coordinates import SkyCoord
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from tqdm import tqdm
from joblib import Parallel, delayed
from pretty import pretty_file
from simbad import xmatch_simbad
from gaia import xmatch_gaia

# pretty_file("final_catalog.csv")

def designation(ra, dec):
    ra = ra % 360
    deg_sgn = "+" if dec >= 0 else "-"
    coord = SkyCoord(ra, abs(dec), unit="deg", frame="icrs")
    ra_hms = coord.ra.hms
    dec_hms = coord.dec.dms

    ra_h = str(int(np.trunc(ra_hms.h))).zfill(2)
    ra_m = str(int(np.trunc(ra_hms.m))).zfill(2)
    ra_s = "{:.2f}".format(ra_hms.s).zfill(5)

    dec_h = str(int(np.trunc(dec_hms.d))).zfill(2)
    dec_m = str(int(np.trunc(dec_hms.m))).zfill(2)
    dec_s = "{:.1f}".format(np.trunc(dec_hms.s*10)/10).zfill(4)
    
    designator = "VarWISE"

    des = designator + " J" + ra_h + ra_m + ra_s + deg_sgn + dec_h + dec_m + dec_s
    return des

def format_ra(ra):
    ra = ra % 360
    ra = (np.round(ra, 5))
    return ra
def format_dec(dec):
    dec = (np.round(dec, 4))
    return dec
def format_mag(mag):
    mag = (np.round(mag, 2))
    return mag

def get_amp(band="w1"):
    def get_amp_inner(flux):
        flux = np.array(flux)
        flux = flux[~np.isnan(flux)]
        flux = flux[flux > 0]
        if len(flux) <= 1:
            if band=="w1":
                print(flux)
                raise Exception("No fluxmag found")
            return
        if band == "w1":
            mag = -2.5 * np.log10(flux * 0.00000154851985514 / 309.54)
        elif band == "w2":
            mag = -2.5 * np.log10(flux * 0.00000249224248693 / 171.787)
        
        i80r = np.quantile(mag, 0.8) - np.quantile(mag, 0.2)
        return np.round(i80r, 3)
    return get_amp_inner

def get_mag(band="w1"):
    def get_mag_inner(flux):
        flux = flux[~np.isnan(flux)]
        flux = flux[flux > 0]
        if len(flux) < 2:
            if band=="w1":
                print(flux)
                raise Exception("No fluxmag found")

            return
        
        if band == "w1":
            mag = -2.5 * np.log10(flux * 0.00000154851985514 / 309.54)
        elif band == "w2":
            mag = -2.5 * np.log10(flux * 0.00000249224248693 / 171.787)
        
        return np.round(np.mean(mag), 3)
    return get_mag_inner

def get_var_snr(row):
    flux = np.array(row["w1flux"])[row["filter_mask"]]
    sig = np.array(row["w1sigflux"])[row["filter_mask"]]

    noise = np.median(sig)
    i90r = np.quantile(flux, 0.9) - np.quantile(flux, 0.1)
    var_snr = i90r / noise
    return np.round(var_snr, 3)

def format_period(period):
    period = (np.round(period, 8))
    return period

def format_period_significance(period_significance):
    period_significance = (np.round(period_significance, 3))
    return period_significance

def format_likelihood(likelihood):
    likelihood = (np.round(likelihood, 4))
    return likelihood

def blendflag(row):
    # If all of W1mag, e_W1mag, W2mag, e_W2mag are missing, then the object is blended
    if pd.isna(row["W1mag"]) and pd.isna(row["e_W1mag"]) and pd.isna(row["W2mag"]) and pd.isna(row["e_W2mag"]):
        if row["type"] == 1:
            return 0 # Indicates that there are no nearby AllWISE sources, and is likely a SN
        return 1 # Indicates that there are no nearby AllWISE sources, and is likely blended
    return 0

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode="same")
    return y_smooth

def periodsignificance(row):
    width = 0.1 # similar to 10 bins
    p = row["period"]
    if not p > 0:
        return
    t = np.array(row["mjd"])[row["filter_mask"]]
    y = np.array(row["w1flux"])[row["filter_mask"]]
    ynorm = (y - np.median(y)) / (np.quantile(y, 0.8) - np.quantile(y, 0.2))

    # Null hyp
    smoothed = smooth(ynorm, int(len(ynorm) * width))
    mse_null = np.mean((ynorm - smoothed) ** 2)

    # Alt hyp
    phase = (t % p) / p
    sorter = np.argsort(phase)
    phase = phase[sorter]
    ynorm = ynorm[sorter]
    smoothed = smooth(ynorm, int(len(ynorm) * width))
    mse_alt = np.mean((ynorm - smoothed) ** 2)

    improvement = mse_null / mse_alt

    return np.round(improvement, 4)


def extract_classification(row):
    if row["type"] == 1:
        if row["extragalactic"]:
            return "sn"
        else:
            return "cv"
    elif row["type"] == 2:
        if row["extragalactic"]:
            return "agn"
        else:
            types = ["ea", "ew", "lpv", "rot", "rr", "cep", "yso"]
            return types[int(row["tree_prediction"])]
    return "Unknown"

def write_catalog(pid):
    try:
        flag_tbl = pd.read_csv(f"/home/mpaz/neovar/inference/flagtbls/partition_{pid}_flag_tbl.csv").set_index("cluster_id")
    except:
        print("No flag table found for partition ", pid)
        return

    try:
        param_tbl = pd.read_csv(f"/home/mpaz/neovar/secondary/subclassifier/inference/out/partition_{pid}.csv").set_index("cluster_id")
    except:
        print("No param table found for partition ", pid)
        return

    try:
        cids = flag_tbl.reset_index()["cluster_id"]
        lc_data = pq.read_table(f"/home/mpaz/neowise-clustering/clustering/out/partition_{pid}_cluster_id_to_data.parquet",
                            filters=pc.is_in(pc.field("cluster_id"), pa.Array.from_pandas(cids)), use_threads=False).to_pandas().set_index("cluster_id")
    except:
        print("No light curve data found for partition ", pid)
        return

    data = flag_tbl.join(param_tbl, how="left", validate="1:1", rsuffix="_r")
    data = data.join(lc_data.drop(["ra", "dec"], axis=1), how="left", validate="1:1")
    catalog = pd.DataFrame(index=data.index)

    blended = data.apply(blendflag, axis=1)
    catalog["Designation"] = data.apply(lambda row: designation(row["ra"], row["dec"]), axis=1)
    catalog["RAJ2000"] = data["ra"].apply(format_ra)
    catalog["DecJ2000"] = data["dec"].apply(format_dec)
    catalog["W1mag"] = data["w1flux"].apply(get_mag("w1"))
    catalog["W2mag"] = data["w2flux"].apply(get_mag("w2"))
    catalog["W3mag"] = data["W3mag"].apply(format_mag)
    catalog["W4mag"] = data["W4mag"].apply(format_mag)
    catalog["Jmag"] = data["Jmag"].apply(format_mag)
    catalog["Hmag"] = data["Hmag"].apply(format_mag)
    catalog["Kmag"] = data["Kmag"].apply(format_mag)
    catalog["W1_amp"] = data["w1flux"].apply(get_amp("w1"))
    catalog["W2_amp"] = data["w2flux"].apply(get_amp("w2"))
    catalog["variability_snr"] = data.apply(get_var_snr, axis=1)
    # catalog["period"] = data["period"].apply(format_period)
    catalog["period_peak_1"] = data["peak1"].apply(format_period)
    catalog["period_peak_2"] = data["peak2"].apply(format_period)
    catalog["period_peak_3"] = data["peak3"].apply(format_period)

    catalog["period_significance"] = data.apply(periodsignificance, axis=1)
    catalog["blended?"] = blended
    catalog["saturated"] = catalog["W1mag"].apply(lambda x: 1 if x < 7.5 else 0)
    catalog["confidence"] = data["tree_score"].apply(format_likelihood)
    catalog["type"] = data.apply(extract_classification, axis=1)

    # FILTERING #
    catalog = catalog[catalog["saturated"] == 0]
    catalog.drop("saturated", axis=1, inplace=True)

    return catalog

# cats = []
# for i in tqdm(range(12288)):
#     print("Writing partition ", i)
#     cats.append(write_catalog(i))
cats = Parallel(n_jobs=-1)(delayed(write_catalog)(i) for i in range(0,12288))

fname = "VarWISE.csv"
FINAL_CATALOG = pd.concat(cats)
# xmatches
simbad_map = xmatch_simbad(FINAL_CATALOG)
FINAL_CATALOG["simbad_type"] = simbad_map

gaia_map = xmatch_gaia(FINAL_CATALOG)
FINAL_CATALOG = FINAL_CATALOG.join(gaia_map, how="left")

FINAL_CATALOG.to_csv("catalogs/"+fname) 
pretty_file("catalogs/"+fname)

def purify(row): # UP FOR REVIEW
    if row["blended?"] == 1:
        return False
    if row["variability_snr"] < 4.75 and not (row["type"] in ["sn", "cv"]):
        return False
    
    if row["type"] in ["sn", "cv"] and row["variability_snr"] < 4.25:
        return False
    
    if row["confidence"] < 0.8 and row["variability_snr"] < 8:
        return False
    
    if row["type"] == "rot" and (row["confidence"] < 0.95 or row["period_significance"] < 1.1):
        return False
    
    if row["W1mag"] > 15.5:
        return False
    return True

print("Initial count: ", len(FINAL_CATALOG))
filter = FINAL_CATALOG.apply(purify, axis=1)
filtered = FINAL_CATALOG[filter]
filtered.drop("blended?", axis=1, inplace=True)
print("Final count: ", len(filtered))

filtered.to_csv("catalogs/pure_"+fname)
pretty_file("catalogs/pure_"+fname)