import torch
import numpy as np
import plotly.graph_objects as go
from genset import GenSet
import os
import shutil
import plotly.io as pio
import sys

sys.path.append("../lib")

many = 25
inst = GenSet(randseed=0)

def plot(ex, savestr, name=""):
    ex = ex.detach().cpu().numpy()
    fig = go.Figure()

    x = ex[:,2]
    y = ex[:,0]
    yerr = ex[:,1]

    tr = go.Scatter(x=x, y=y, mode="markers", marker=dict(size=6, opacity=.8, color="black"), name="Data", marker_symbol="square", error_y=dict(type="data", array=yerr, visible=True))

    fig.add_trace(tr)
    # set background to flat white with no gridlines
    fig.update_layout(
        font=dict(
            family="monospace",
            size=7,
            color="Black"
        ),
        plot_bgcolor="white",
        # margin=dict(l=0, r=0, t=0, b=0) # remove margin
    )

    # remove legend, add black border and remove tick marks
    fig.update_layout(showlegend=False, width=800, height=600, title=name)
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True, showticklabels=True, ticks="", showgrid=False)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True, showticklabels=True, ticks="", showgrid=False)
    # set font size to 1

    
    pio.write_image(fig, f"./imgs/{savestr}.png")


# clear the directory
shutil.rmtree("./imgs")
os.mkdir("./imgs")

# for i in range(many):
#     i = str(i)
#     ex = inst.gen_stochastic()
#     plot(ex, "stochastic" + i)

for i in range(50):
    i = str(i)
    ex, params = inst.gen_nova(True)
    # list params in title
    title = "Nova: "
    for key in params:
        title += f"{key}={params[key]}, "
    plot(ex, "nova" + i, name=title)

for i in range(5):
    i = str(i)
    ex = inst.gen_null()
    plot(ex, "null" + i)

for i in range(5):
    i = str(i)
    ex, pd = inst.gen_transit(returnpd=True)
    plot(ex, "transit" + i)
    ex[:,2] = ex[:,2] % pd
    plot(ex, "transitfolded" + i)

for i in range(5):
    i = str(i)
    ex, pd, ph = inst.gen_pulsating_var(returnpd=True)
    plot(ex, "pulsating" + i)
    ex[:,2] = (ex[:,2] - ph) % pd
    plot(ex, "pulsatingfolded" + i, name="{}".format(pd))


    



