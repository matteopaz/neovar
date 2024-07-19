import plotly.graph_objects as go
import pandas as pd
import numpy as np

valid_data = pd.read_parquet('valid_data.parquet')
print(valid_data[["source_id", "type"]])

for objrow in valid_data.iterrows():
    objrow = objrow[1]
    t = objrow["mjd"]
    y = objrow["w1flux"]
    y = np.array(y) * 0.00000154851985514
    y = - 2.5 * np.log10(y / 309.54)

    tr = go.Scatter(x=t, y=y, mode='markers')
    fig = go.Figure(tr)
    fig.update_yaxes(title_text="W1 Magnitude", autorange="reversed")
    fig.update_xaxes(title_text="MJD")

    name = str(objrow["type"]) + "_" + str(objrow["source_id"])
    fig.update_layout(title=name)
    fig.write_image("./vis/{}.png".format(name))
