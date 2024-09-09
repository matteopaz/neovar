from lib import write_as_catalog, w1f_to_mag, w1sf_to_sigmpro
import pandas as pd
import numpy as np

def classify_periodic(table): # change the table type column and return a catalog
    periodics = table.loc[table["type"] == "periodic"]

    if not periodics.empty: 
        newtypes = periodics.apply(classify_single_periodic, axis=1)
        periodics.loc[:, "type"] = newtypes
        table.loc[table["type"] == "periodic", "type"] = newtypes
        periodics = periodics[periodics["type"].notna()]
    return write_as_catalog(periodics)

def classify_single_periodic(row):
    t = row["mjd"]
    flux = row["w1flux"]
    sigflux = row["w1sigflux"]

    avg_flux = np.mean(flux) # Multiply by DN conversion value if using validation / filter optimization
    avg_mag = -2.5 * np.log10(avg_flux / 309.54)


    if len(t) < 90:
        return pd.NA
    
    iqr = np.quantile(flux, 0.8) - np.quantile(flux, 0.15)
    if iqr < 3 * np.median(sigflux):
        return pd.NA
    
    # if avg_mag < 7.75: # Sensor saturation (Breaks for validation as flux is the wrong format)
    #     return pd.NA
    
    return "variable"
    
    

    