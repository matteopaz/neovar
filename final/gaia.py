from astropy import table
from astropy import units as u
import pandas as pd
from astroquery.xmatch import XMatch
import numpy as np

def xmatch_gaia(catalog):
    
    COLS = ["Source", "Plx", "e_Plx", "Dist", "Gmag", "e_Gmag", "BPmag", "e_BPmag", "RPmag", "e_RPmag", "A0", "AG", "E(BP-RP)"]
    G_absmag = lambda Gmag, Dist: Gmag - 5 * (np.log10(Dist) - 1)
    varwise_tbl = table.Table.from_pandas(catalog.reset_index())[["cluster_id", "RAJ2000", "DecJ2000"]]
    cat = "vizier:I/355/gaiadr3" # Gaia DR3
    max_radius = 2
    NPARTS = len(catalog) // 150000 + 1
    res_parts = []

    for n in range(NPARTS):
        results = XMatch.query(cat1=varwise_tbl[n*150000:(n+1)*150000], cat2=cat, max_distance=max_radius * u.arcsec, colRA1='RAJ2000', colDec1='DecJ2000')
        res = results.to_pandas().set_index("cluster_id")
        res = res[COLS]
        res = res.sort_values("Gmag")
        res = res.groupby(res.index).first()
        res["G_absmag"] = G_absmag(res["Gmag"], res["Dist"])
        res_parts.append(res)

    final = pd.concat(res_parts)
    final.drop_duplicates(inplace=True)
    return final