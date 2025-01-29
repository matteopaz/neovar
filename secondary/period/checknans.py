import pandas as pd
import os
import numpy as np

ptd = list(range(12288 * 13//20, 12288 * 14//20))

for partition_id in ptd:
    try:
        data = pd.read_parquet("./cached/partition_{}.parquet".format(partition_id))
    except:
        continue
    hasnan = lambda x: np.isnan(x).any()
    magnan = data["mag"].apply(hasnan)
    timenan = data["time"].apply(hasnan)
    print("Partition {} has {} NaNs in mag and {} NaNs in time".format(partition_id, magnan.sum(), timenan.sum()))

