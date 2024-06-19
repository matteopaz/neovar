import pyarrow
import pyarrow.compute
import pickle

base_path = "/stage/irsa-data-parquetlinks/links-tmp/neowiser/healpix_k5/"
year_path = "year<N>_skinny/neowiser-healpix_k5-year<N>_skinny.parquet/_metadata"
neowise_path = lambda year: base_path + year_path.replace("<N>", str(year))
year_datasets = [
pyarrow.dataset.parquet_dataset(neowise_path(year), partitioning="hive") for year in range(1, 11)
]

neowise_ds = pyarrow.dataset.dataset(year_datasets)
pickle.dump(neowise_ds, open("neowise_ds.pkl", "wb"))