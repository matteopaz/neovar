import astroquery
from astroquery.ipac.irsa import Irsa
import plotly.graph_objects as go
import astropy
import astropy.units as u
import pandas as pd

# center = "246.3658542 -23.4898612"
center = "266.4059312 -28.9345025"

Irsa.ROW_LIMIT = 1000000
tbl = Irsa.query_region(center, catalog='neowiser_p1bs_psd', spatial="Cone",
                            radius=12 * 10 * u.arcsec, columns="ra,dec,w1mpro,w1sigmpro,cc_flags,w1rchi2,saa_sep,w1snr,qual_frame,moon_masked,sso_flg")

tbl = tbl.to_pandas()

tbl.to_csv("transient_test.csv")

tbl = Irsa.query_region(center, catalog='unwise_2019', spatial="Cone",
                            radius=12 * 10 * u.arcsec, columns="ra,dec,flux_1")

tbl = tbl.to_pandas()

tbl.to_csv("transient_test_truth.csv")