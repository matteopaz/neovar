import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyarrow as pa
import pyarrow.parquet as pq

TYPE = "rot"
training_data = pq.read_table("/home/mpaz/neovar/secondary/data/training_data.parquet", filters=pa.compute.equal(pa.compute.field("type"), TYPE)).to_pandas()
print(len(training_data))
training_data = training_data[training_data["type"] == TYPE]

# periods = np.array(training_data["W3mag"].values) - np.array(training_data["W1mag"].values)
# periods = periods[~np.isnan(periods)]
periods = np.array(training_data["peak1"].values)
len(periods)
periods = [np.log10(p) for p in periods if p > 0]
len(periods)

# make histogram of periods

hist, bins = np.histogram(periods, bins=180)

plt.bar(bins[:-1], hist, width=np.diff(bins))
plt.savefig("periodhist.png")

# print(training_data.iloc[1010][["ra", "dec", "peak1"]])