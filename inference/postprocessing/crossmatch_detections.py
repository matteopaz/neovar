import pandas as pd
import numpy as np
from astropy import units as u
from astroquery.xmatch import XMatch
from astropy.table import Table
import os
from joblib import Parallel, delayed
allwise_cat = "vizier:II/328/allwise"

QUERY_CHUNKSIZE = 512000
asec_radius = 1
input_tables = []
for i in range(12288)[1::2]:
    try:
        tbl = pd.read_csv(f"/home/mpaz/neovar/inference/out2_filtered/partition_{i}_flag_tbl.csv")
    except:
        print(f"Partition {i} not found")
    tbl["ORIGINAL_PARTITION"] = i
    input_tables.append(tbl)

bigtbl = pd.concat(input_tables)
bigtbl["ra"] = bigtbl["ra"] % 360
print(len(bigtbl))

def query(chunk_i, cat, cols=[]):
    chunk = Table.from_pandas(bigtbl.iloc[chunk_i*QUERY_CHUNKSIZE:(chunk_i+1)*QUERY_CHUNKSIZE])
    query = XMatch.query(cat1=chunk,
                        cat2=cat,
                        max_distance=asec_radius * u.arcsec, colRA1='ra',
                        colDec1='dec')
    tbl = query.to_pandas()
    tbl = tbl[["cluster_id", "ra", "dec", "type", "confidence", "extragalactic", "ORIGINAL_PARTITION", "angDist"] + cols]
    return tbl

cols = ["W1mag", "e_W1mag", "W2mag", "e_W2mag", "W3mag", "e_W3mag", "W4mag", "e_W4mag", "Jmag", "e_Jmag", "Hmag", "e_Hmag", "Kmag", "e_Kmag", "var", "ex"]
chunk_responses_allwise = Parallel(n_jobs=4)(delayed(query)(i, allwise_cat, cols) for i in range(len(bigtbl) // QUERY_CHUNKSIZE + 1))
allwise_result = pd.concat(chunk_responses_allwise).set_index("cluster_id")
allwise_result = allwise_result.reset_index().sort_values("angDist").drop_duplicates(subset="cluster_id", keep="first").set_index("cluster_id")

all_objs = bigtbl.set_index("cluster_id")
for col in cols:
    all_objs[col] = allwise_result[col]

partition_grouped = all_objs.groupby("ORIGINAL_PARTITION")

for pid, group in partition_grouped:
    out = group.drop(columns=["ORIGINAL_PARTITION"])
    out.to_csv(f"/home/mpaz/neovar/inference/out2_filtered/partition_{pid}_flag_tbl.csv")
