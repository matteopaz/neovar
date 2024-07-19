import pandas as pd
import numpy as np
import warnings

def write_as_catalog(table):
    if len(table) == 0:
        return pd.DataFrame(columns=["designation", "ra", "dec", "w1mpro", "w2mpro", "period", "type", "confidence"])
    tbl = pd.DataFrame({"designation": table["designation"], 
                        "ra": None, "dec": None, "w1mpro": None, "w2mpro": None, 
                        "period": None, "type": None, "confidence": None, "cluster_id": table["cluster_id"]})

    tomag_w1 = lambda flux: np.where((flux > 0) & ~np.isnan(flux), -2.5 * np.log10(flux / 309.54), np.nan)
    tomag_w2 = lambda flux: np.where(flux / 171.787 > 0, -2.5 * np.log10(flux / 171.787), np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tbl["w1mpro"] = table["w1flux"].apply(tomag_w1)
        tbl["w2mpro"] = table["w2flux"].apply(tomag_w2)

    tbl["w1mpro"] = tbl["w1mpro"].apply(np.nanmedian)
    tbl["w2mpro"] = tbl["w2mpro"].apply(np.nanmedian)
    tbl["type"] = table["type"]

    tbl["confidence"] = table["conf"]
    
    centroids = [get_centroid(row["ra"], row["dec"]) for _, row in table.iterrows()]
    tbl["ra"] = [centroid[0] for centroid in centroids]
    tbl["dec"] = [centroid[1] for centroid in centroids]

    if "period" in table.columns:
        tbl["period"] = table["period"]

    return tbl

def get_centroid(ra, dec):
    ra = np.radians(ra)
    dec = np.radians(dec)
    cartesian = np.array([np.cos(ra) * np.cos(dec), np.sin(ra) * np.cos(dec), np.sin(dec)])
    centroid = np.mean(cartesian, axis=1)
    centroid_ra = np.arctan2(centroid[1], centroid[0]) * 180 / np.pi
    centroid_dec = np.arcsin(centroid[2]) * 180 / np.pi

    return centroid_ra, centroid_dec