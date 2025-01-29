import torch
import pandas as pd
import numpy as np
import pickle
import sys
sys.path.append("/home/mpaz/neovar/inference")
from model import WCNFourierModel
from sklearn import metrics
import plotly.graph_objects as go

# def evaluate(true, pred):
#     f1 = metrics.f1_score(true, pred, average="macro")
#     precision = metrics.precision_score(true, pred, average="macro")
#     recall = metrics.recall_score(true, pred, average="macro")

#     transient_true = true[true !=2]
#     transient_pred = pred[true !=2]
#     transient_true = transient_true == 1
#     transient_pred = transient_pred == 1
#     transient_precision = metrics.precision_score(transient_true, transient_pred, average="macro")
#     transient_recall = metrics.recall_score(transient_true, transient_pred, average="macro")

#     periodic_true = true[true !=1]
#     periodic_pred = pred[true !=1]
#     periodic_true = periodic_true == 2
#     periodic_pred = periodic_pred == 2
#     periodic_precision = metrics.precision_score(periodic_true, periodic_pred, average="macro")
#     periodic_recall = metrics.recall_score(periodic_true, periodic_pred, average="macro")

#     return {"f1": f1, "precision": precision, "recall": recall, "transient_precision": transient_precision, "transient_recall": transient_recall, "periodic_precision": periodic_precision, "periodic_recall": periodic_recall}



torch.set_default_device("cuda")
modelname = "newvalid_best"
params = pickle.load(open(f"/home/mpaz/neovar/inference/model/{modelname}.pkl", "rb"))
model = WCNFourierModel(**params).cuda()
model.load_state_dict(torch.load(f"/home/mpaz/neovar/inference/model/{modelname}.pt"))
model.eval().cuda()
BATCHSIZE = 384

valid = pickle.load(open("/home/mpaz/neovar/training/valid/valid_data.pkl", "rb"))
tens = valid[0]
label = valid[1]
preds = model(tens)
probs = torch.nn.functional.softmax(preds, dim=1)

maxes, l1_classes = torch.max(probs, dim=1)

maxes = maxes.detach().cpu().numpy()
true = label.detach().cpu().numpy().argmax(axis=1)
pred = l1_classes.detach().cpu().numpy()

conf = 0.9

confidence_cut = pred.copy()
confidence_cut[maxes <= conf] = 0

matr = metrics.confusion_matrix(true, confidence_cut)
print(matr)
print(metrics.classification_report(true, confidence_cut))


# f1s = []
# tps = []
# trs = []
# pps = []
# prs = []
# precision = []
# recall = []

# samples = np.linspace(0.5, 1, 500) - 0.001

# true[true == 2] = 1

# # for conf in samples:
# #     confidence_cut = pred.copy()
# #     confidence_cut[maxes <= conf] = 0
# #     confidence_cut[confidence_cut == 2] = 1


# #     res = evaluate(true, confidence_cut)
# #     precision.append(res["precision"])
# #     recall.append(res["recall"])

# # f1tr = go.Scatter(x=samples, y=f1s, name="F1")
# # tptr = go.Scatter(x=samples, y=tps, name="Transient Precision")
# # trtr = go.Scatter(x=samples, y=trs, name="Transient Recall")
# # pptr = go.Scatter(x=samples, y=pps, name="Periodic Precision")
# # prtr = go.Scatter(x=samples, y=prs, name="Periodic Recall")
# prec = go.Scatter(x=samples, y=precision, name="Precision")
# rec = go.Scatter(x=samples, y=recall, name="Recall")

# fig = go.Figure([prec, rec])
# # fig = go.Figure([f1tr, tptr, trtr, pptr, prtr])
# fig.update_layout(title="Metrics vs Confidence Threshold", xaxis_title="Confidence Threshold", yaxis_title="Metric Value")
# fig.write_image("conf.png")
# fig.write_html("conf.html")

