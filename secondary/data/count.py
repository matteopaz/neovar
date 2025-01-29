import pandas as pd

a = pd.read_parquet("training_data.parquet")

print(len(a))