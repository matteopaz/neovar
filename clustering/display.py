import plotly.graph_objects as go
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pickle as pkl
from tqdm import tqdm

partition_id = 0
 

neowise_ds = pkl.load(open("neowise_ds.pkl", "rb"))
# data_tbl = pd.read_csv(f"./out/partition_{partition_id}_source_id_to_cluster_id.csv")
map_tbl = pq.read_table(f"./out/partition_{partition_id}_cluster_id_to_source_id.parquet").to_pandas()
print("stuff loaded")

all_indices = []
for i in range(len(map_tbl)):
    all_indices += map_tbl.iloc[i]["source_ids"]

partition_filter = pc.equal(pc.field("healpix_k5"), partition_id)
source_id_filter = pc.field("source_id").isin(all_indices)
tbl = neowise_ds.to_table(filter=partition_filter & source_id_filter, columns=["ra", "dec", "source_id", "healpix_k5"])

traces = []
progress = tqdm(total=len(map_tbl))
for i in range(len(map_tbl)):
    progress.update(1)
    indices = map_tbl.iloc[i]["source_ids"]

    partition_filter = pc.equal(pc.field("healpix_k5"), partition_id)
    source_id_filter = pc.field("source_id").isin(indices)


    traces.append(go.Scatter(
        x=tbl.["ra"], 
        y=tbl["dec"], 
        mode='markers', marker=dict(size=2.75), 
    ))
fig = go.Figure(data=traces)
# size = 2000 x 2000
fig.update_layout(width=2000, height=2000)
fig.write_image(f"./out/partition_{partition_id}_clusters.png")


    
    


