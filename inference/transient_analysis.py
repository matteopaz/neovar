from lib import write_as_catalog
import numpy as np
import pandas as pd
 
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


def classify_single_transient(row):
    t = row["mjd"]
    flux = row["w1flux"]
    sigflux = row["w1sigflux"]
    conf = row["confidence"]

    if conf < 0.925:
        return pd.NA
    
    if len(t) < 20:
        return pd.NA
    
    epoch_t, epoch_f, epoch_sf = get_epochs(row)
    max_epoch_flux, max_epoch_n = np.max(np.array([np.median(f) for f in epoch_f])), np.argmax(np.array([np.median(f) for f in epoch_f]))
    max_epoch_sigflux = np.median(epoch_sf[max_epoch_n])

    if np.abs(max_epoch_flux - np.median(flux)) < 3 * np.median(sigflux): # 3 sigma above the measured noise
        return pd.NA
    
    if max_epoch_flux / max_epoch_sigflux < 5: # Transient should be above 5 sigma
        return pd.NA
    

    
    return "transient"

