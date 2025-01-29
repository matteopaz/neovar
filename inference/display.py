import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from load_data import PartitionDataLoader

# Initialize the Dash app
app = dash.Dash(__name__)

# Load initial data
partition = 816
datatbl = pd.read_parquet(f"out/partition_{partition}_subcatalog.parquet")
pdl = PartitionDataLoader(datatbl, len(datatbl))
data_iter = iter(pdl)

# Function to create plot from tensor
def plot(row):
    yflux = row["w1flux"].astype(float)
    ysigflux = row["w1sigflux"].astype(float)
    t = row["mjd"].astype(float)

    y = -2.5 * np.log10(yflux / 309.54)
    yerr = y + 2.5 * np.log10((yflux - ysigflux) / 309.54)

    tr = go.Scatter(x=t, y=y, error_y=dict(type='data', array=yerr, visible=True, width=0.15, color="rgba(0,0,0,0.4)"), 
                    mode='markers', marker_symbol="square", marker_size=3.75, opacity=0.9)
    fig = go.Figure(tr)
    name = row["designation"]
    type = row["type"]
    conf = row["confidence"]
    fig.update_layout(title=f"{name} ({type} @ {conf:.4f})", width=1200, height=800)
    fig.update_yaxes(autorange="reversed")
    return fig

# Define the layout of the app
app.layout = html.Div([
    dcc.Graph(id='plot-graph'),
    html.Button('Next Plot', id='next-plot-button', n_clicks=0),
    html.Div(id='coordinates', style={'margin-top': '20px'})
])

# Define the callback to update the plot and coordinates
@app.callback(
    [Output('plot-graph', 'figure'),
     Output('coordinates', 'children')],
    [Input('next-plot-button', 'n_clicks')]
)
def update_plot(n_clicks):
    objn = n_clicks
    row = datatbl.iloc[objn]
    fig = plot(row)
    ra = np.mean(row["ra"])
    dec = np.mean(row["dec"])
    coordinates = f"{ra}, {dec}"
    return fig, coordinates

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')