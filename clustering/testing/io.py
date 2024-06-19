import numpy as np
import pyarrow
import pyarrow.compute
import hpgeom
import pandas as pd
import pyarrow.dataset
import matplotlib.pyplot as plt

def show(db):
    if isinstance(db, pyarrow.Table):
        db = db.to_pandas()
    ra = db["ra"].to_numpy() - np.pi
    dec = db["dec"].to_numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='mollweide')
    ax.scatter(ra, dec, s=1.4)
    plt.show()


# Creation of dummy dataset
def dummy_sphere(n_pts):
    N = n_pts
    A = 4 * np.pi / N # Area corresponding to any point. Implied by equidistance
    D = np.sqrt(A) # side length of rhomboids on sphere centered at each point
    n_latitude_points = np.round(np.pi / D)

    x = []
    y = []

    for dec in np.linspace(-np.pi/2, np.pi/2, int(n_latitude_points)):
        n_longitude_points = np.ceil(2 * np.pi * np.cos(dec) / D)
        for ra in np.linspace(0, 2*np.pi, int(n_longitude_points)):
            x.append(ra)
            y.append(dec)
    return x,y

ra,dec = dummy_sphere(10000)
pandas_tb = pd.DataFrame({'ra':ra, 'dec':dec})
pyarrow_tb = pyarrow.Table.from_pandas(pandas_tb)


ORDER = 13
NSIDE = hpgeom.order_to_nside(ORDER)

healpix_field = hpgeom.hpgeom.angle_to_pixel(NSIDE, pyarrow_tb["ra"], pyarrow_tb["dec"])

print(healpix_field)
show(pyarrow_tb)

