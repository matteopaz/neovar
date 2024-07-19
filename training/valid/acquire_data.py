import pandas as pd
import os
import pyarrow as pa
import pyarrow.dataset
import pyarrow.compute as pc
import hpgeom as hpg
from joblib import Parallel, delayed


base_path = "/stage/irsa-data-parquet10/wise/neowiser/p1bs_psd/healpix_k5/" 
year_path = "year<N>_skinny/neowiser-healpix_k5-year<N>_skinny.parquet/_metadata"
neowise_path = lambda year: base_path + year_path.replace("<N>", str(year))

year_datasets = [
pa.dataset.parquet_dataset(neowise_path(year), partitioning="hive") for year in range(1, 11)
]

neowise_ds = pa.dataset.dataset(year_datasets)

catalog = pd.concat([pd.read_csv("./sheets/{}".format(x)) for x in os.listdir("./sheets") if x.endswith(".csv")])
catalog.reset_index(drop=True, inplace=True)

def query(ra, dec):
    rad = 3 / 3600
    radsq = rad ** 2

    rafield = pc.field('ra')
    decfield = pc.field('dec')

    radist = pc.subtract(rafield, ra)
    decdist = pc.subtract(decfield, dec)
    distsq = pc.add(pc.power(radist, 2), pc.power(decdist, 2))
    distfilter = pc.less(distsq, radsq)

    pixel = hpg.angle_to_pixel(32, ra, dec)
    pixelfield = pc.field('healpix_k5')
    pixelfilter = pc.equal(pixelfield, pixel)

    tbl = neowise_ds.to_table(filter=(distfilter & pixelfilter), columns = [
            "cntr", "ra", "dec",
            "mjd", "w1flux", "w1sigflux", "w2flux", "w2sigflux",
            "qual_frame", "w1rchi2", "w2rchi2"
        ]).to_pandas()
    
    tbl["cluster_id"] = 0
    
    return tbl

# data = Parallel(n_jobs=8)(delayed(query)(ra, dec) for ra, dec in zip(catalog["ra"], catalog["dec"]))
data = []
for i, (ra, dec) in enumerate(zip(catalog["ra"], catalog["dec"])):
    print("{} out of {}".format(i, len(catalog)))
    data.append(query(ra, dec))

data_dict = {"ra": [], "dec": [], "mjd": [], "w1flux": [], "w1sigflux": [], "w2flux": [], "w2sigflux": [], "qual_frame": [], "w1rchi2": [], "w2rchi2": []}
for tbl in data:
    for key in data_dict.keys():
        data_dict[key].append(tbl[key].values)

datacatalog = pd.DataFrame(data_dict)

print(catalog["source_id"])
print(catalog["type"])

datacatalog["source_id"] = catalog["source_id"]
datacatalog["type"] = catalog["type"]

datacatalog["source_id"] = datacatalog["source_id"].astype(str)
datacatalog["type"] = datacatalog["type"].astype(str)

datacatalog.to_parquet("valid_data.parquet")
