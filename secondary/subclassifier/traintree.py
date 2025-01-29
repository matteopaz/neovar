import torch
from dataloader import TreeDL
from varnet import VARnet
from autoencoder import Morphologic
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report
import pickle

def sensible_filters(datatbl):
    # EWs
    ew = (datatbl["type"] == "ew") & (datatbl["period"] < 1.5)
    # EAs
    ea = (datatbl["type"] == "ea") & (datatbl["period"] > 0.5)
    # CEPs
    cep = (datatbl["type"] == "cep") & (datatbl["period"] > 1) & (datatbl["period"] < 75)
    # RRs
    rr = (datatbl["type"] == "rr") & (datatbl["period"] < 2)
    ## LPVs
    lpv = (datatbl["type"] == "lpv") & (datatbl["period"] > 50)
    ## YSOs - Reddening
    yso = (datatbl["type"] == "yso")
    yso_color = (datatbl["W1mag"] - datatbl["W2mag"] > -1) | (datatbl["W1mag"] - datatbl["W3mag"] > 0) # shouldnt be significantly blued
    yso = yso & yso_color
    ## Rotators
    rot = (datatbl["type"] == "rot") & (datatbl["period"] < 10) # according to https://arxiv.org/pdf/2206.05505
    ## RS Canum Venaticorum-type stars
    # rscvn = (datatbl["type"] == "rscvn") & (datatbl["period"] < 14)

    return ew | ea | cep | rr | lpv | yso | rot 

traindata = pd.read_parquet("/home/mpaz/neovar/secondary/data/training_data.parquet")
print(len(traindata))
# model = VARnet(512, 8, wavelet="bior2.2", learnsamples=True, infeatures=2)
model = Morphologic(64, 7, features=2)
model.load_state_dict(torch.load("/home/mpaz/neovar/secondary/subclassifier/model/morpho.pth"))

tdl = TreeDL(traindata, model, training=True, lc_bins=64, filters=sensible_filters)
train, valid, test = tdl.train_valid_test()

train.to_parquet("train_featuretbl.parquet")
valid.to_parquet("valid_featuretbl.parquet")
test.to_parquet("test_featuretbl.parquet")

print("Training data: ", len(train))
print("Validation data: ", len(valid))
print("Test data: ", len(test))

dtrain = xgb.DMatrix(train.drop("type", axis=1), label=train["type"].values)
dvalid = xgb.DMatrix(valid.drop("type", axis=1), label=valid["type"].values)
dtest = xgb.DMatrix(test.drop("type", axis=1), label=test["type"].values)

dtrain.save_binary("dtrain.dmatrix")
dvalid.save_binary("dvalid.dmatrix")
dtest.save_binary("dtest.dmatrix")


# print("Training XGBoost model")

# tree = xgb.XGBClassifier(n_estimators=30, max_depth=6, learning_rate=0.1, n_jobs=8, verbosity=2, objective="multi:softmax", num_class=8)

# tree.fit(train.drop("type", axis=1), train["type"].values, eval_set=[(valid.drop("type", axis=1), valid["type"].values)])

# # print classification report
# pred_y = tree.predict(test.drop("type", axis=1))
# true_y = test["type"].values
# print(classification_report(true_y, pred_y))
