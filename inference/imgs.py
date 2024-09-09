import pandas as pd
import plotly.graph_objects as go
import numpy as np
from lib import w1f_to_mag, w1sf_to_sigmpro
from display import plot

partition = 8977

subcatalog = pd.read_parquet(f"out/partition_{partition}_table.parquet")

plots = subcatalog.apply(plot, axis=1)

for i, p in enumerate(plots):
    p.write_image(f"imgs/{subcatalog.iloc[i]['designation']}.png")