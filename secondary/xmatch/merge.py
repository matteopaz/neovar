import os
import pandas as pd

cats = pd.concat([pd.read_csv(f"/home/mpaz/neovar/secondary/xmatch/results/{f}") for f in os.listdir('./results/') if f.endswith('.csv')], ignore_index=True)
cats.to_csv('xmatched.csv', index=False)

grouped = cats.groupby('cluster_id')
print(len(grouped), ' matches')