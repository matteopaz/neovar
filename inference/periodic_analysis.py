from lib import write_as_catalog, w1f_to_mag, w1sf_to_sigmpro, get_centroid
import pandas as pd
import numpy as np
import astropy.units as u
import astropy.coordinates as coord

def classify_periodic(table): # change the table type column and return a catalog
    periodics = table.loc[table["type"] == "periodic"]

    if not periodics.empty: 
        newtypes = periodics.apply(classify_single_periodic, axis=1)
        periodics.loc[:, "type"] = newtypes
        table.loc[table["type"] == "periodic", "type"] = newtypes
        periodics = periodics[periodics["type"].notna()]
    return write_as_catalog(periodics)

def get_galactic_dec(row):
    ra, dec = get_centroid(row["ra"], row["dec"])
    coordinates = coord.SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    galactic = coordinates.transform_to(coord.Galactic())
    return galactic.b.degree

def get_periodic_analysis_features(row):
    t = row["mjd"]
    flux = row["w1flux"]
    sigflux = row["w1sigflux"]
    galdec = np.abs(get_galactic_dec(row))
    galmod = max(0, (10 - galdec)/10)
    avg_flux = np.mean(flux) # Multiply by DN conversion value if using validation / filter optimization
    avg_mag = -2.5 * np.log10(avg_flux / 309.54)
    i8r = np.quantile(flux, 0.8) - np.quantile(flux, 0.15)
    iqr = np.quantile(flux, 0.75) - np.quantile(flux, 0.25)
    
    return {"galdec": galdec, "avg_flux": avg_flux, "avg_mag": avg_mag, "i8r_to_median": i8r / np.median(flux), "iqr_to_median": iqr/np.median(flux)}

def classify_single_periodic(row):
    t = row["mjd"]
    flux = row["w1flux"]
    sigflux = row["w1sigflux"]
    galdec = np.abs(get_galactic_dec(row))
    galmod = max(0, (10 - galdec)/10)

    avg_flux = np.mean(flux) # Multiply by DN conversion value if using validation / filter optimization
    avg_mag = -2.5 * np.log10(avg_flux / 309.54)

    if len(t) < 120:
        return pd.NA 
    
    i8r = np.quantile(flux, 0.8) - np.quantile(flux, 0.15)
    
    if i8r < (3 + 2*galmod) * np.median(sigflux):
        return pd.NA
    
    iqr = np.quantile(flux, 0.75) - np.quantile(flux, 0.25)
    
    if avg_mag < 8 and iqr < (2 / avg_mag): # Sensor saturation (Breaks for validation as flux is the wrong format)
        return pd.NA
    
    return "variable"
    
    
