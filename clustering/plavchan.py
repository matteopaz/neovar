from sklearn.metrics.pairwise import haversine_distances
from math import radians
bsas = [-34.83333, -58.5166646 + 180]
paris = [49.0083899664, 2.53844117956 + 180]
bsas_in_radians = [radians(_) for _ in bsas]
paris_in_radians = [radians(_) for _ in paris]
result = haversine_distances([bsas_in_radians, paris_in_radians])
print(result)