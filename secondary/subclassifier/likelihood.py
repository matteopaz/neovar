import numpy as np
import pandas as pd
import os
from sklearn.mixture import BayesianGaussianMixture

TYPE = "ew"

typenum = ["ea", "ew", "lpv", "rot", "rr", "cep", "rscvn", "yso"].index(TYPE)

data = pd.concat([pd.read_csv(f"inference/out/{f}") for f in os.listdir("inference/out/") if f.endswith(".csv")])

oftype = data[data["tree_prediction"] == typenum]
oftype = oftype[["W1mag", "W2mag", "W3mag", "W4mag", "Jmag", "Hmag", "Kmag", "period", 
                 "period_significance", "confidence", "skew", "kurt",
                 "feature_0", "feature_1", "feature_2", "feature_3", "feature_4",
                 "feature_5", "feature_6", "feature_7"]]

mix = BayesianGaussianMixture(n_components=1, max_iter=1000)

mix.fit(oftype)

print(mix.means_)