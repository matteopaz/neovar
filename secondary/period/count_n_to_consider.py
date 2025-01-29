import pandas as pd
import os

c = [int(f.split('_')[1]) for f in os.listdir('./out') if f.endswith('.csv')]

allp = [int(f.split('_')[1]) for f in os.listdir('/home/mpaz/neovar/inference/out2_filtered/') if f.endswith('.csv')]
allc = [int(f.split('_')[1][:-8]) for f in os.listdir('./cached') if f.endswith('.parquet')]

print(set(allp) - set(allc))

print(set() - set(c))
print(len(allp - set(c)))