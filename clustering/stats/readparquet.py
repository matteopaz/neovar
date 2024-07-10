import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
from hpgeom import hpgeom as hpg

# Read /home/mpaz/neowise-clustering/clustering/out/partition_0_cluster_id_to_data.parquet
table = pq.read_table('/home/mpaz/neowise-clustering/clustering/out/partition_10101_cluster_id_to_data.parquet')
df = table.to_pandas()

clusterone = df.iloc[1010]
cid = int(clusterone['cluster_id'])

ra = np.array(clusterone['ra'])
dec = np.array(clusterone['dec'])

# convert to cartesian and take the centroid
x = np.cos(np.radians(dec)) * np.cos(np.radians(ra))
y = np.cos(np.radians(dec)) * np.sin(np.radians(ra))
z = np.sin(np.radians(dec))

centroid = np.array([np.mean(x), np.mean(y), np.mean(z)])
ra, dec = hpg.vector_to_angle(centroid, degrees=True)
ra = ra[0]
dec = dec[0]

# round to 4
centroid_ra = round(ra, 4)
centroid_dec = round(dec, 4)
binreo = np.binary_repr(int(centroid_ra * 10000), width=24)
print(binreo)
print(int((binreo), 2))

print("Centered at ra, dec:", centroid_ra, centroid_dec)

print(bin(cid))
print(f"Partition ID: {(cid & int('1'*16 + '0'*48, 2)) >> 48}")

binra = ((cid & int("0"*16 + "1"*24 + "0"*24, 2)) >> 24) / 10000
bindec = np.round(90 - (cid & int("0"*40 + "1"*24, 2)) / 10000, 4)
print(binra, bindec)
