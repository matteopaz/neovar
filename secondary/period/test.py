from get_period import get_period
import pandas as pd
import os
from time import perf_counter as pc

tbls = [pd.read_csv("./highsample.csv")]
tbls = [tbl.dropna(subset=["w1mpro", "mjd"], how="any") for tbl in tbls]
horizontal_tbl = pd.DataFrame({"time": [tbl["mjd"].values for tbl in tbls]*1024, "mag": [tbl["w1mpro"].values for tbl in tbls]*1024})
print(list(f for f in os.listdir("./") if f.endswith(".csv")))
start = pc()
periods = get_period(horizontal_tbl)
end = pc()
print((end - start)/100)
# print(periods)