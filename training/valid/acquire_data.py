import pandas as pd
import os
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import pyarrow.compute as pc
import hpgeom as hpg
from joblib import Parallel, delayed

catalog = pd.concat([pd.read_csv("./sheets/{}".format(x)) for x in os.listdir("./sheets") if x.endswith(".csv")])
catalog.reset_index(drop=True, inplace=True)

catalog["pix"] = catalog.apply(lambda x: hpg.angle_to_pixel(32, x["ra"], x["dec"]), axis=1)

def haversine_dist(ra, dec, id=pc.field("cluster_id")):

    radians = lambda x: pc.multiply(x, 3.14159265358979323846 / 180)
    ra2 = pc.multiply(ra, 3.14159265358979323846 / 180)
    dec2 = pc.multiply(dec, 3.14159265358979323846 / 180)

    ra1 = pc.divide(pc.bit_wise_and(pc.shift_right(id, 24), int("1"*24, 2)), 10000.0)
    dec1 = pc.add(pc.multiply(pc.divide(pc.bit_wise_and(id, int("1"*24, 2)),10000.0), -1), 90)
    ra1 = pc.multiply(ra1, 3.14159265358979323846 / 180) # convert to radians
    dec1 = pc.multiply(dec1, 3.14159265358979323846 / 180)

    lon_sin2 = pc.power(pc.sin(pc.divide(pc.subtract(ra1, ra2), 2)), 2)
    cos_lon_terms = pc.multiply(pc.cos(dec1), pc.cos(dec2))
    inner_term_2 = pc.multiply(lon_sin2, cos_lon_terms)

    lat_sin2 = pc.power(pc.sin(pc.divide(pc.subtract(dec1, dec2), 2)), 2)

    inner_term = pc.sqrt(pc.add(lat_sin2, inner_term_2))

    return pc.multiply(pc.asin(inner_term), 2)

def euc_dist(ra, dec, id=pc.field("cluster_id")):
    ra2 = pc.multiply(ra, 3.14159265358979323846 / 180)
    dec2 = pc.multiply(dec, 3.14159265358979323846 / 180)

    ra1 = pc.divide(pc.bit_wise_and(pc.shift_right(id, 24), int("1"*24, 2)), 10000.0)
    dec1 = pc.add(pc.multiply(pc.divide(pc.bit_wise_and(id, int("1"*24, 2)),10000.0), -1), 90)
    ra1 = pc.multiply(ra1, 3.14159265358979323846 / 180) # convert to radians
    dec1 = pc.multiply(dec1, 3.14159265358979323846 / 180)

    lon_diff = pc.subtract(ra1, ra2)
    lat_diff = pc.subtract(dec1, dec2)

    return pc.sqrt(pc.add(pc.power(lon_diff, 2), pc.power(lat_diff, 2)))

def query(ra, dec) -> pd.DataFrame:
    rad = (150/3600) * 3.1415 / 180

    pix = hpg.angle_to_pixel(32, ra, dec)

    distfilter = pc.less(euc_dist(ra, dec), rad)

    tbl = pq.read_table(
        '/home/mpaz/neowise-clustering/clustering/out/partition_{}_cluster_id_to_data.parquet'.format(pix),
        filters=distfilter).to_pandas()
    
    # add column generated by haversine_dist
    d = np.array([haversine_dist(ra, dec, id=x).as_py() for x in tbl["cluster_id"]])
    tbl["haversine_dist"] = d * (180/3.14159265) * 3600
    tbl.sort_values("haversine_dist", inplace=True, ascending=True)

    if len(tbl) == 0:
        print(Warning("No data found for this source: ra={}, dec={}".format(ra, dec)))
        return
    
    return tbl.iloc[0]

data = []
for i in range(len(catalog)):
    print("Querying source {} of {}".format(i, len(catalog)))
    tbl = query(catalog["ra"][i], catalog["dec"][i])
    data.append(tbl)
    

data_dict = {"ra": [], "dec": [], "mjd": [], "w1flux": [], "w1sigflux": [], "w2flux": [], "w2sigflux": [], "qual_frame": [], "w1rchi2": [], "w2rchi2": []}
for tbl in data:
    for key in data_dict.keys():
        if tbl is not None:
            data_dict[key].append(tbl[key])
        else:
            data_dict[key].append(np.array([]))

datacatalog = pd.DataFrame(data_dict)

print(catalog["source_id"])
print(catalog["type"])

datacatalog["source_id"] = catalog["source_id"].astype(str)
datacatalog["type"] = catalog["type"].astype(str)
datacatalog["period"] = catalog["period"].astype(float)

datacatalog.to_parquet("valid_data.parquet")
