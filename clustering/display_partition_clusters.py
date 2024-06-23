import plotly.graph_objects as go
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pickle as pkl
from tqdm import tqdm

partition_id = 0

data = pq.read_table(f"./out/partition_{partition_id}_cluster_id_to_data.parquet").to_pandas()

traces = []

progress = tqdm(total=len(data))
i = 0
for row in data.iterrows():
    if i == 10000:
        break
    i += 1
    progress.update(1)
    row = row[1]
    ra = row["ra"]
    dec = row["dec"]
    cid = row["cluster_id"]

    trace = go.Scatter(
        x=ra,
        y=dec,
        mode="markers",
        marker=dict(
            size=3,
            color=cid,
        )
    )

    traces.append(trace)

fig = go.Figure(data=traces)
fig.write_image(f"partition_{partition_id}_clusters.png")




    
    


