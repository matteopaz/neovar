import numpy as np 
import pandas as pd
import pickle as pkl
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

# append /home/mpaz/neovar/inference to sys.path
import sys
sys.path.append('/home/mpaz/neovar/inference')
from transient_analysis import classify_single_transient
from periodic_analysis import classify_single_periodic

valid = pd.read_parquet('/home/mpaz/neovar/training/valid/valid_data.parquet')

transients = valid.loc[valid["type"].isin(("sn", "nova")), :]
periodics = valid.loc[~valid["type"].isin(("sn", "nova", "const")), :]
nulls = valid.loc[valid["type"] == "const", :]

transients["secondary"] = transients.apply(classify_single_transient, axis=1)
periodics["secondary"] = periodics.apply(classify_single_periodic, axis=1)
nulls["transient_secondary"] = nulls.apply(classify_single_transient, axis=1)
nulls["pulsator_secondary"] = nulls.apply(classify_single_periodic, axis=1)


label_typemap ={
    "const": 0,
    "nova": 1,
    "sn": 1,
    "yso": 2,
    "agn": 2,
    "ceph": 2,
    "rvt": 2,
    "rr": 2,
    "mira": 2,
    "sr": 2,
    "ea": 2,
    "ew": 2,
    "mix": 2
}

pred_typemap = {
    "transient": 1,
    "variable": 2,
    pd.NA: 0
}

transients["type"] = transients["type"].map(label_typemap)
transients["secondary"] = transients["secondary"].map(pred_typemap)
periodics["type"] = periodics["type"].map(label_typemap)
periodics["secondary"] = periodics["secondary"].map(pred_typemap)
nulls["type"] = nulls["type"].map(label_typemap)


nullpulsators = nulls.loc[nulls["pulsator_secondary"] == "variable", :]
nullpulsators["secondary"] = 2
nulltransients = nulls.loc[nulls["transient_secondary"] == "transient", :]
nulltransients["secondary"] = 1
truenulls = nulls.loc[nulls["transient_secondary"].isna() & nulls["pulsator_secondary"].isna(), :]
truenulls["secondary"] = 0

transient_analysis = pd.concat((transients, nulltransients, truenulls))
true_transient = np.array(transient_analysis["type"].values)
pred_transient = np.array(transient_analysis["secondary"].values)

pulsator_analysis = pd.concat((periodics, nullpulsators, truenulls))
true_pulsator = np.array(pulsator_analysis["type"].values)
pred_pulsator = np.array(pulsator_analysis["secondary"].values)


print("TRANSIENT:")
print(confusion_matrix(true_transient, pred_transient))
print(classification_report(true_transient, pred_transient))
print(f1_score(true_transient, pred_transient, average="weighted"))
print("PULSATOR:")
print(confusion_matrix(true_pulsator, pred_pulsator))
print(classification_report(true_pulsator, pred_pulsator))
print(f1_score(true_pulsator, pred_pulsator, average="weighted"))

# transient_tp = len(transients.loc[transients["secondary"] == "transient"]) 
# transient_fp = len(nulls.loc[nulls["transient_secondary"] == "transient"])
# transient_actual = len(transients)
# pulsator_tp = len(periodics.loc[periodics["secondary"] == "variable"]) 
# pulsator_fp = len(nulls.loc[nulls["pulsator_secondary"] == "variable"])
# pulsator_actual = len(periodics)
# nulls_tp = len(nulls.loc[nulls["transient_secondary"].isna() & nulls["pulsator_secondary"].isna()]) 
# nulls_fp = len(transients.loc[transients["secondary"].isna()]) + len(periodics.loc[periodics["secondary"].isna()])
# nulls_actual = len(nulls)   

# print("Transients predicted: ", transient_tp + transient_fp)
# print("Transient FPs: ", transient_fp)
# print("Transient FNs: ", transient_actual - transient_tp)

# print("Periodics predicted: ", pulsator_tp + pulsator_fp)
# print("Pulsator FPs: ", pulsator_fp)
# print("Pulsator FNs: ", pulsator_actual - pulsator_tp)

# print("Nulls predicted: ", nulls_tp + nulls_fp)
# print("Nulls FPs: ", nulls_fp)
# print("Nulls FNs: ", nulls_actual - nulls_tp)

# transient_recall = transient_tp / transient_actual
# transient_precision = transient_tp / (transient_tp + transient_fp)
# pulsator_recall = pulsator_tp / pulsator_actual
# pulsator_precision = pulsator_tp / (pulsator_tp + pulsator_fp)
# nulls_recall = nulls_tp / nulls_actual
# nulls_precision = nulls_tp / (nulls_tp + nulls_fp)


# print("Transient recall: ", transient_recall)
# print("Transient precision: ", transient_precision)
# print("Pulsator recall: ", pulsator_recall)
# print("Pulsator precision: ", pulsator_precision)
