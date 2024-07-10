import healpy as hp
import json
import numpy as np
import matplotlib.pyplot as plt
from hpgeom import hpgeom as hpg

cmap = list(json.load(open("./cluster_sizes.json", "r")))

clusters = [0] * 12288

angles = [hpg.pixel_to_angle(32, i, nest=False, lonlat=True, degrees=True) for i in range(12288)]
hp_pix = [hp.ang2pix(32, *angle, lonlat=True, nest=False) for angle in angles]

for i, pix in enumerate(hp_pix):
    clusters[pix] = cmap[i]

clusters = np.array(clusters)

print(clusters[::900])
print(np.sum(clusters))

hp.cartview(clusters, nest=False, title="Density map of sources across the sky")
plt.savefig("./greatmosaic.png")