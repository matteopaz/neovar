import pandas as pd
import os
from joblib import Parallel, delayed

def process(file):
    try:
        pid = int(file.split("_")[1])
        original = pd.read_csv("/home/mpaz/neovar/inference/out2_filtered/" + file).set_index("cluster_id")
        periods = pd.read_csv(f"/home/mpaz/neovar/secondary/period/out/partition_{pid}_periods.csv").set_index("cluster_id")

        joined = original.join(periods, how="left", validate="one_to_one")

        joined.to_csv(f"/home/mpaz/neovar/inference/flagtbls/partition_{pid}_flag_tbl.csv", index=True)
    except Exception as e:
        print(e)
        print(f"Error in {file}")
        return
    
Parallel(n_jobs=16)(delayed(process)(file) for file in os.listdir("/home/mpaz/neovar/inference/out2_filtered/"))

    