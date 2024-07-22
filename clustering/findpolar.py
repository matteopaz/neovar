from hpgeom import hpgeom as h

# find all order 5 pixels that include points 0.5 degrees away from (90, 66.54) or (-90, 66.54)

import numpy as np
from hpgeom import hpgeom as h

x_center = 270
y_center = 66.54
rad = 0.212

x = np.linspace(x_center - rad, x_center + rad, 250)
y = [[np.sqrt(rad**2 - (xi - x_center)**2) + y_center, -np.sqrt(rad**2 - (xi - x_center)**2) + y_center] for xi in x]
y = np.array(y).flatten()
x = np.array([xi for xi in x for _ in range(2)])
pix = h.angle_to_pixel(32, x, y)
pix = set(pix)

print(pix)

