import astroquery
from astroquery.ipac.irsa import Irsa
import plotly.graph_objects as go
import astropy
import astropy.units as u
from astropy.coordinates import SkyCoord
import pandas as pd
Irsa.ROW_LIMIT = 1000000

# center = "246.3658542 -23.4898612"

coord = lambda ra, dec: SkyCoord(ra, dec, unit=(u.deg, u.deg), frame="icrs")

side_length = 150 * u.arcsec
ctrs = [coord(246.3695, -23.4853), coord(266.4071, -28.9342), coord(50.00, 50.00), coord(90, -66.56)]
# idx 0 is nebulous region in the Rho Ophiuchi Cloud complex, featuring dense spurious nebulous detections and artifacts from nearby bright stars
# idx 1 is the galactic center
# idx 2 is a region in the Perseus constellation (Ra and Dec 50.00) representative of the majority of the sky. More isolated sources and reliable detections
# idx 3 is centered roughly about the south ecliptic pole, a region with very high sampling frequency


for i, ctr in enumerate(ctrs):
    print(f"Querying region {i}...")
    tbl = Irsa.query_region(ctr, catalog='neowiser_p1bs_psd', spatial="Box",
                                width=side_length, columns="ra,dec,w1mpro,w1sigmpro,w1cc_map,w1rchi2,saa_sep,w1snr,qual_frame,moon_masked,sso_flg")
    tbl = tbl.to_pandas()

    tbl.to_csv(f"region{i}.csv")

    tbl = Irsa.query_region(ctr, catalog='unwise_2019', spatial="Box",
                                width=side_length, columns="ra,dec,flux_1")

    tbl = tbl.to_pandas()

    tbl.to_csv(f"region{i}_truth.csv")