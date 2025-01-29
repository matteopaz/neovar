import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

valid = pd.read_parquet('/home/mpaz/neovar/training/valid/valid_data.parquet')

null_idxs = valid["type"] == "const"

nulls = valid.loc[null_idxs, :]
variables = valid.loc[~null_idxs, :]

# features

def gen_features(row):
    w1 = row["w1flux"]
    w1s = row["w1sigflux"]
    w2 = row["w2flux"]
    w2s = row["w2sigflux"]

    if np.nanmean(w1) / np.nanmean(w1s) > np.nanmean(w2) / np.nanmean(w2s):
        y = w1
        ys = w1s
    else:
        y = w2
        ys = w2s

    fix = ~np.isnan(y)
    y = y[fix]
    ys = ys[fix]

    noise = np.median(ys)

    median_ = np.median(y)
    mad = np.median(np.abs(y - median_))
    median_skew = np.median((y - median_) ** 3) / mad ** 3

    mad_norm = mad / noise

    i60r = (np.percentile(y, 60) - np.percentile(y, 40)) / noise
    iqr = (np.percentile(y, 75) - np.percentile(y, 25)) / noise
    i85r = (np.percentile(y, 85) - np.percentile(y, 15)) / noise

    return np.array([mad_norm, median_skew, i60r, iqr, i85r])

Xv = np.stack(variables.apply(gen_features, axis=1).values)
Yv = np.ones(Xv.shape[0])
Xn = np.stack(nulls.apply(gen_features, axis=1).values)
Yn = np.zeros(Xn.shape[0])

X = np.concatenate([Xv, Xn])
Y = np.concatenate([Yv, Yn])

tree = DecisionTreeClassifier(max_depth=1)
tree.fit(X, Y)

Yp = tree.predict(X)
print(confusion_matrix(Y, Yp))
# display the tree
from sklearn.tree import export_text
r = export_text(tree, feature_names=["mad_norm", "median_skew", "i60r", "iqr", "i85r"])
print(r)