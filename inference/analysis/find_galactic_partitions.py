import astropy
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
import hpgeom.hpgeom as hpg

xr = np.linspace(-90, 90, 300)
yr = np.linspace(-8, 8, 32)
pts = np.array(np.meshgrid(xr, yr)).T.reshape(-1, 2)

# convert from gal to equatorial
c = SkyCoord(pts[:, 0] * u.deg, pts[:, 1] * u.deg, frame='galactic').transform_to('icrs')
# get partitions at nside 32
nside = 32
pixels = set(hpg.angle_to_pixel(nside, c.ra.deg, c.dec.deg))

print(pixels)