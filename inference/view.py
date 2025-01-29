import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np

loaded_partitions = {}

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    dcc.Input(id='input-box', type='text', value=''),
    html.Button('Submit', id='button'),
    dcc.Graph(id='output-graph')
])

# Define the callback function
@app.callback(
    Output('output-graph', 'figure'),
    [Input('button', 'n_clicks')],
    [dash.dependencies.State('input-box', 'value')]
)

def update_graph(_, value):
    cid = int(value)
    partition = cid >> 48
    if partition not in loaded_partitions:
        loaded_partitions[partition] = pd.read_parquet(
            "/home/mpaz/neowise-clustering/clustering/out/partition_{}_cluster_id_to_data.parquet".format(partition)).set_index("cluster_id")
    data = loaded_partitions[partition].loc[cid]

    y = -2.5 * np.log10(data["w1flux"]*0.00000154851985514 / 309.54)
    yerr = data["w1sigflux"]
    x = data["mjd"]
    ra = np.mean(data["ra"])
    dec = np.mean(data["dec"])

    tr = go.Scatter(x=x, y=y, mode='markers', marker_symbol="square", marker_size=3.15, opacity=0.7)
    fig = go.Figure(tr)
    fig.update_layout(title=f"Cluster {cid} at {ra}, {dec}", width=1200, height=800)
    fig.update_yaxes(autorange="reversed")
    return fig


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)