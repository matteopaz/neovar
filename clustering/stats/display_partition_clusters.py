import plotly.graph_objects as go
from bokeh.plotting import figure, output_file, save
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from hpgeom import hpgeom

# coords = (267.18, 66.44)
# rad = 100 / 3600
# pix = hpgeom.angle_to_pixel(32, coords[0], coords[1], degrees=True)
# print(pix)

PARTITION_ID = 2901

# PARTITION_ID = 7594
# N_CLUSTERS_TO_PLOT = 400

# Load the partition
path_to_outfiles = "/home/mpaz/neowise-clustering/clustering/out/"
datafile_path = path_to_outfiles + "partition_{}_cluster_id_to_data.parquet".format(PARTITION_ID)

tbl = pq.read_table(datafile_path).to_pandas()
print(len(tbl))

ra = tbl["ra"].values
dec = tbl["dec"].values
ids = tbl["cluster_id"].values

fig = figure(title="Partition {}".format(PARTITION_ID), x_axis_label="RA", y_axis_label="Dec", width=1920, height=1080)

i = 0
for idx, ra_list, dec_list, cid in zip(range(len(tbl)), ra, dec, ids):
    # plot only every 6th point
    cntrra = np.mean(ra_list)
    cntrdec = np.mean(dec_list)

    # if (cntrra - coords[0])**2 + (cntrdec - coords[1])**2 > rad**2:
    #     continue
    
    w1 = np.median(tbl.iloc[idx]["w1flux"])
    w1 = 20.752 - 2.5 * np.log10(w1)
    
    ra_list = ra_list[::10]
    dec_list = dec_list[::10]
    ra_list = np.array(ra_list)
    dec_list = np.array(dec_list)
    fig.scatter(x=ra_list, y=dec_list, legend_label=f"Cluster {cid}", size=1)

output_file(filename="partition_{}_plot_bokeh.html".format(PARTITION_ID))
save(fig)

