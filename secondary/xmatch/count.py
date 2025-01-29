import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

tbl = pd.read_csv("xmatched.csv")
count = tbl.value_counts("type")
print(count)
print(len(tbl))

# print(tbl[tbl["type"] == "rot"].sort_values(by="dist").head(20))
# print(tbl[tbl["type"] == "oscillator"].sort_values(by="dist").head(20))
# print value count

print(tbl["type"].value_counts())
print(tbl[tbl["type"] == "WD"])