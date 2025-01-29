import numpy as np
import pandas as pd
import os
# from sklearn.mixture import BayesianGaussianMixture
import corner


types = ["ea", "ew", "lpv", "rot", "rr", "cep", "rscvn", "yso"]

for TYPE in types:
    typenum = types.index(TYPE)
    data = pd.concat([pd.read_csv(f"inference/out/{f}") for f in os.listdir("inference/out/") if f.endswith(".csv")])

    oftype = data[data["tree_prediction"] == typenum]

    X = oftype[["W2mag", "W3mag", "skew", "i75r", "kurt", "period", "period_significance", "confidence"]].dropna(how="any").to_numpy()

    print(len(X))

    ranges = [(np.quantile(X[:,i], 0.925), np.quantile(X[:,i], 0.075)) for i in range(X.shape[1])]

    plot = corner.corner(X, labels=["W2mag", "W4mag", "skew", "i75r", "kurt", "period", "period_significance", "confidence"], show_titles=True,
                        range=ranges)
    plot.savefig(f"corner_{TYPE}.png")

