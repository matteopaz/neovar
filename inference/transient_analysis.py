from lib import write_as_catalog, get_centroid
import numpy as np
import pandas as pd
import astropy.units as u
import astropy.coordinates as coord
 
def classify_transient(table): # change the table type column and return a catalog
    transients = table.loc[table["type"] == "transient"]
    newtypes = transients.apply(classify_single_transient, axis=1)
    transients.loc[:, "type"] = newtypes
    table.loc[table["type"] == "transient", "type"] = newtypes
    transients = transients[transients["type"].notna()]
    return write_as_catalog(transients)

def get_epochs(row):
    t = row["mjd"]
    dt = np.concatenate((np.zeros(1), np.diff(t)))
    epochskip = np.where(dt > 10)[0]
    epochlen = np.diff(np.concatenate((epochskip, [len(t)])))
    epoch_n_threshold = epochlen > 7
    epochskip = epochskip[epoch_n_threshold]

    te = np.split(t, epochskip)
    fe = np.split(row["w1flux"], epochskip)
    se = np.split(row["w1sigflux"], epochskip)
    return te, fe, se

def zigscore(row):
    row["w1flux"] = (row["w1flux"] - np.nanmedian(row["w1flux"])) / (np.nanquantile(row["w1flux"], 0.9) - np.nanquantile(row["w1flux"], 0.1))
    epochs_t, epochs_f, epochs_s = get_epochs(row)

    epoch_f = np.array([np.nanmedian(e) for e in epochs_f])
    if np.isnan(epoch_f).any():
        return 0

    if len(epochs_f) < 2:
        return 0

    dp = np.diff(epoch_f)
    pattern = np.zeros_like(dp)
    pattern[dp > 0] = 1
    pattern[dp < 0] = -1

    up_triangles = 0

    for i in range(len(pattern) - 1):
        if pattern[i] == 1 and pattern[i+1] == -1:
            up_triangles += 1

    if len(pattern) < 2:
        return 0

    score = up_triangles / (len(pattern) // 2)
    return score

def varscore(row):
    row["w1flux"] = (row["w1flux"] - np.nanmedian(row["w1flux"])) / (np.nanquantile(row["w1flux"], 0.9) - np.nanquantile(row["w1flux"], 0.1))
    epochs_t, epochs_f, epochs_s = get_epochs(row)

    if len(epochs_f) < 2:
        return 1
    
    north_scan_dir = np.array([np.nanmedian(e) for e in epochs_f[0::2]])
    south_scan_dir = np.array([np.nanmedian(e) for e in epochs_f[1::2]])

    overall = np.array([np.nanmedian(e) for e in epochs_f])   

    if np.isnan(overall).any():
        return 1

    if np.isnan(north_scan_dir).any():
        row["w1flux"] = row["w1flux"] + 1

    northvar = np.nanvar(north_scan_dir)
    southvar = np.nanvar(south_scan_dir)
    separated_var = np.nanvar(overall)
    if np.isnan(northvar):
        separated_var = southvar 
    elif np.isnan(southvar):
        separated_var = northvar
    else:
        separated_var = min(northvar, southvar)

    score = separated_var / np.nanvar(overall)
    return score

def get_galactic_dec(row):
    ra, dec = get_centroid(row["ra"], row["dec"])
    coordinates = coord.SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    galactic = coordinates.transform_to(coord.Galactic())
    return galactic.b.degree

def classify_single_transient(row):
    t = row["mjd"]
    flux = row["w1flux"]
    sigflux = row["w1sigflux"]
    galdec = np.abs(get_galactic_dec(row))
    galmod = max(0, (10 - galdec)/10)

    if len(t) < 20:
        return pd.NA
    
    epoch_t, epoch_f, epoch_sf = get_epochs(row)
    max_epoch_flux, max_epoch_n = np.max(np.array([np.median(f) for f in epoch_f])), np.argmax(np.array([np.median(f) for f in epoch_f]))
    max_epoch_sigflux = np.median(epoch_sf[max_epoch_n])

    zig = zigscore(row) # latent flagging
    var = varscore(row)

    if (zig > 0.8 and var < 0.3) or (var < 0.1):
        return pd.NA

    if np.abs(max_epoch_flux - np.median(flux)) < (3+galmod*3) * np.median(sigflux): # 3 sigma above the measured noise
        return pd.NA
    
    if max_epoch_flux / max_epoch_sigflux < 5 + galmod*5: # Transient should be above 5 sigma
        return pd.NA
    
    return "transient"

