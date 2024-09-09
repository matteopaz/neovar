import numpy as np
import plotly.graph_objects as go
import pandas as pd
import pickle
import torch

torch.set_default_device('cpu')
datatens, _, _, names = pickle.load(open('valid_data.pkl', 'rb'))

name = "Nova Sgr 2021a"
name = "Nova Men 2022"

idx = names.tolist().index(name)
data = datatens[idx].cpu().numpy()

# plot data[:, 0] vs data[:, 2] with data[:, 1] as error bars
tr = go.Scatter(x=data[:, 2], y=data[:, 0], mode='markers', marker=dict(size=4, color="blue"), error_y=dict(type='data', array=data[:, 1], visible=True, width=0.125, color="gray"), marker_symbol="square")
fig = go.Figure(tr)
fig.update_yaxes(title_text="Adj Magnitude")
fig.update_xaxes(title_text="Adj Timestamp")
fig.write_image("view.png")