import pandas as pd
from joblib import Parallel, delayed

def getcat(i):
    try:
        return pd.read_csv(f'out2_filtered/partition_{i}_flag_tbl.csv')
    except:
        return None

cats = Parallel(n_jobs=4)(delayed(getcat)(i) for i in range(12288))
cats = [cat for cat in cats if cat is not None]
cats = pd.concat(cats)
print(len(cats))
print(len(cats[cats["type"] == 1]))
print(len(cats[cats["type"] == 2]))