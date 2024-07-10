import numpy as np
import sklearn
import sklearn.metrics
import matplotlib.pyplot as plt
import healpy

havdist = sklearn.metrics.pairwise.haversine_distances

# create a list of points with x ranging from -180 to 180 and y ranging from -90 to 90

order = 5
nside = 2**order
n_pix = healpy.nside2npix(nside)

errs = [0] * n_pix
pts = np.array([healpy.pix2ang(nside, i, lonlat=True) for i in range(n_pix)])
pts_compare = pts + 0.85/3600

flipped_pts = np.array([pts[:,1], pts[:,0]]).T
flipped_pts_compare = flipped_pts + 0.85/3600


# calculate the pairwise distances
distances_true = havdist(np.radians(flipped_pts), np.radians(flipped_pts_compare))
distances_false = havdist(np.radians(pts), np.radians(pts_compare))

distances_true = (distances_true).diagonal()
distances_false = (distances_false).diagonal()

err = np.abs(distances_true - distances_false) / distances_true
print(distances_true)

print(len(err))

# healpy.visufunc.cartview(err, title="Error in distance calculation", cbar=True)
# plt.savefig("error_in_distance_calculation.png")

err_mod = np.array([10 if i > 0.1 else 0 for i in err])
healpy.visufunc.cartview(err_mod, title="Error in distance calculation above 10%", cbar=True)
plt.savefig("error_in_distance_calculation_above_10.png")
