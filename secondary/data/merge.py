import pandas as pd
import os

cats = [pd.read_parquet(f) for f in os.listdir(".") if f.endswith(".parquet")]
data = pd.concat(cats)
print(len(cats[0]))
data.to_parquet("training_data.parquet")