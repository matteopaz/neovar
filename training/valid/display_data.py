import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

valid_data = pd.read_parquet('valid_data.parquet')
print(valid_data[["source_id", "type"]])

traces = []
titles = []
getrowcol = lambda i: (i // 3 + 1, i % 3 + 1)

i = 0
for objrow in valid_data.iterrows():
    objrow = objrow[1]
    # if objrow["type"] not in ["cep", "sr", "mira", "ew", "ea", "rvt", "const"]:
    #     continue
    t = objrow["mjd"]
    w1f = objrow["w1flux"] * 0.00000154851985514
    w1s = objrow["w1sigflux"] * 0.00000154851985514
    w1mag = -2.5 * np.log10(w1f / 309.54)
    w1magerr = -2.5 * np.log10((w1f + w1s) / 309.54) - w1mag

    # remove 3 sigma outliers
    idxr = np.abs(w1mag - np.mean(w1mag)) < 3.5 * np.std(w1mag)
    t = t[idxr]
    w1mag = w1mag[idxr]
    w1magerr = w1magerr[idxr]


    tr = go.Scatter(x=t, y=w1mag, mode='markers', marker=dict(size=4, color="blue", ), error_y=dict(type='data', array=w1magerr, visible=True, width=0.125, color="gray"), marker_symbol="square")
    fig = go.Figure(tr)
    fig.update_yaxes(title_text="W1 Magnitude", autorange="reversed")
    fig.update_xaxes(title_text="MJD")

    name = str(objrow["type"]) + "_" + str(objrow["source_id"] + " - {} pts".format(len(w1mag)))
    fig.update_layout(title=name)

    traces.append(tr)
    titles.append(name)
    fig.write_image("./vis/{}.png".format(name))

    if objrow["period"] > 0:
        x_fold = t % objrow["period"]
        tr = go.Scatter(x=x_fold, y=w1mag, mode='markers', marker=dict(size=4, color="blue", ), error_y=dict(type='data', array=w1magerr, visible=True, width=0.125, color="gray"), marker_symbol="square")
        fig = go.Figure(tr)
        fig.update_yaxes(title_text="W1 Magnitude", autorange="reversed")
        fig.update_xaxes(title_text="Phase")
        name = str(objrow["type"]) + "_fold"
        fig.update_layout(title=name)

        traces.append(tr)
        titles.append(name)
        fig.write_image("./vis/{}_fold.png".format(name))
    i += 1

trace_groups = [[t for t in trg] for trg in np.array_split(traces, np.ceil(len(traces) / 12))]
title_groups = [[t for t in trg] for trg in np.array_split(titles, np.ceil(len(titles) / 12))]

pgn = 1
for trgroup, titlegroup in zip(trace_groups, title_groups):
    fig = make_subplots(rows=4, cols=3, subplot_titles=titlegroup)
    fig.update_annotations(font_size=9)
    for i, tr in enumerate(trgroup):
        row, col = getrowcol(i)
        fig.add_trace(tr, row=row, col=col)
        fig.update_yaxes(autorange="reversed", row=row, col=col)
    
    fig.update_layout(width=816, height=1056, title="Page {} / {}".format(pgn, len(trace_groups)))
    # set the title font size to 14
    fig.update_layout(title_font_size=9)
    fig.write_image(f"./vis/pages/{pgn}.pdf", format="pdf")
    pgn += 1