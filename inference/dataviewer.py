import astropy
import plotly.graph_objects as go
from astropy.coordinates import SkyCoord
from astroquery.simbad import Simbad
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import plotly.graph_objects as go
import astrobase as ab
import astrobase.periodbase

catalog_in = pd.read_csv("p0test.csv")

TO_CLUSTER_TABLES = "./in"

def query(cluster_id):
    partition_id = cluster_id >> 48
    partition_table = pq.read_table(f"{TO_CLUSTER_TABLES}/partition_{partition_id}_cluster_id_to_data.parquet", 
                                    filters=pc.equal(pc.field("cluster_id"), cluster_id))
    tbl = partition_table.to_pandas()
    return tbl

def plot(row):
    w1 = row["w1flux"] * 0.00000154851985514
    w1s = row["w1sigflux"] * 0.00000154851985514
    w2 = row["w2flux"] * 0.00000249224248693
    w2s = row["w2sigflux"] * 0.00000249224248693
    mjd = row["mjd"]

    mjd = mjd - np.min(mjd)

    # reject outliers
    non_outliers = np.abs(w1 - np.mean(w1)) < 4.5 * np.std(w1)
    real_det = ~np.isnan(w1s) | ~np.isnan(w2s)

    w1 = w1[non_outliers & real_det]
    w1s = w1s[non_outliers & real_det]
    w2 = w2[non_outliers & real_det]
    w2s = w2s[non_outliers & real_det]
    mjd = mjd[non_outliers & real_det]

    # convert to magnitudes
    w1mag = -2.5 * np.log10(w1 / 309.54)
    w1sigmag = np.abs(w1mag - (-2.5 * np.log10((w1 + w1s) / 309.54)))
    w2mag = -2.5 * np.log10(w2 / 171.787)
    w2sigmag = np.abs(w2mag - (-2.5 * np.log10((w2 + w2s) / 171.787)))

    # fodl
    # bestperiod = ab.periodbase.spdm.stellingwerf_pdm(mjd, w1mag, w1sigmag)["bestperiod"]

    w1tr = go.Scatter(x=mjd, y=w1mag, mode="markers", name="W1", error_y=dict(type="data", array=w1sigmag, visible=True, thickness=0.5, color="rgba(0,0,0,0.3)"), marker=dict(color="blue"))
    w2tr = go.Scatter(x=mjd, y=w2mag, mode="markers", name="W2", error_y=dict(type="data", array=w2sigmag, visible=True, thickness=0.5, color="rgba(0,0,0,0.3)"), marker=dict(color="orange"))
    # w1tr = go.Scatter(x=mjd, y=w1mag, mode="markers", name="W1", error_y=dict(type="data", array=w1sigmag, visible=True), marker=dict(color="blue"))
    # w2tr = go.Scatter(x=mjd, y=w2mag, mode="markers", name="W2", error_y=dict(type="data", array=w2sigmag, visible=True), marker=dict(color="orange"))
    fig = go.Figure([w1tr, w2tr])
    # flip y axis
    fig.update_yaxes(autorange="reversed")

    # w1tr_fold = go.Scatter(x=mjd % bestperiod, y=w1mag, mode="markers", name="W1", error_y=dict(type="data", array=w1sigmag, visible=True, thickness=0.5, color="rgba(0,0,0,0.3)"), marker=dict(color="blue"))
    # w2tr_fold = go.Scatter(x=mjd % bestperiod, y=w2mag, mode="markers", name="W2", error_y=dict(type="data", array=w2sigmag, visible=True, thickness=0.5, color="rgba(0,0,0,0.3)"), marker=dict(color="orange"))
    # fold = go.Figure([w1tr_fold, w2tr_fold])
    fold.update_yaxes(autorange="reversed")

    return fig, fold

for i, row in catalog_in.iterrows():
    if row["w1mpro"] > 14:
        continue
    cluster_id = row["cluster_id"]
    tbl = query(cluster_id)
    fig, fold = plot(tbl.iloc[0])
    fig.update_layout(title=f"{row['designation']} - {row['type']} @ {row['confidence']} confidence", xaxis_title="MJD", yaxis_title="Magnitude")
    fig.show()
    if row["type"] == "EB":
        fold.update_layout(title=f"{row['designation']} - {row['type']} @ {row['confidence']} confidence", xaxis_title="Folded MJD", yaxis_title="Magnitude")
        fold.show() 
    input("Press Enter to continue...")





    