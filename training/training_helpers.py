import plotly.graph_objects as go
from IPython.display import clear_output, display
import sklearn.metrics as metrics
import numpy as np
import pandas as pd


full_typemap = {
    "const": 0,
    "nova": 1,
    "sn": 2,
    "yso": 3,
    "agn": 4,
    "ceph": 5,
    "rvt": 5,
    "rr": 6,
    "mira": 7,
    "sr": 8,
    "ea": 9,
    "ew": 10
}

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
    "mix": None
}

n_types = len(set(full_typemap.values()))
n_classes = len(set(label_typemap.values()))

valid_table = pd.read_parquet("./valid/valid_data.parquet")

def __create_valid_plot__(objrow):
    t = objrow["mjd"]
    w1f = objrow["w1flux"] * 0.00000154851985514
    w1s = objrow["w1sigflux"] * 0.00000154851985514
    w1mag = -2.5 * np.log10(w1f / 309.54)
    w1magerr = -2.5 * np.log10((w1f + w1s) / 309.54) - w1mag

    # remove 3 sigma outliers
    idxr = np.abs(w1mag - np.mean(w1mag)) < 3.5 * np.std(w1mag)
    t = t[idxr]
    w1mag = w1mag[idxr]
    w1magerr = w1magerr[idxr]


    tr = go.Scatter(x=t, y=w1mag, mode='markers', marker=dict(size=4, color="blue"), error_y=dict(type='data', array=w1magerr, visible=True, width=0.125, color="gray"), marker_symbol="square")
    fig = go.Figure(tr)
    fig.update_yaxes(title_text="W1 Magnitude", autorange="reversed")
    fig.update_xaxes(title_text="MJD")

    name = str(objrow["type"]) + "_" + str(objrow["source_id"] + " - {} pts".format(len(w1mag)))
    fig.update_layout(title=name)

    return fig

def training_plot(trainlosses, validations, xrange=100):
    """
    Plots and displays a plotly graph of the training and validation losses, along with a confusion matrix.
    Trainlosses is a list of cross entropy losses for each epoch.
    validation is a dictionary with keys 'true_y', 'pred_y' and 'loss'
    """

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=list(range(len(trainlosses))), y=trainlosses, mode='lines', name='Train loss'))

    f1_scores = [metrics.f1_score(validation["true_y"], validation["pred_y"], average='macro') for validation in validations]
    fig.add_trace(go.Scatter(x=list(range(len(f1_scores))), y=f1_scores, mode='lines', name='Validation F1 score'))

    entropies = [validation["loss"] for validation in validations]
    fig.add_trace(go.Scatter(x=list(range(len(entropies))), y=entropies, mode='lines', name='Validation loss'))

    fig.update_layout(title='Training and validation loss', xaxis_title='Epoch', yaxis_title='Loss')
    fig.update_xaxes(range=[0, xrange])

    conf_matrix = np.zeros((n_types, n_classes))

    for true, pred in zip(validations[-1]["true_type_y"], validations[-1]["pred_y"]):
        conf_matrix[full_typemap[true], pred] += 1
    
    confmatrix_plot = go.Figure()
    hmap = go.Heatmap(z=conf_matrix, x=["Null", "Transient", "Variable"], y=[k for k in label_typemap.keys()], colorscale='blues')
    # add numbers to heatmap
    confmatrix_plot.add_trace(hmap)
    for i in range(len(conf_matrix)):
        for j in range(len(conf_matrix[i])):
            confmatrix_plot.add_annotation(x=j, y=i, text=str(int(conf_matrix[i][j])), showarrow=False)
    confmatrix_plot.update_layout(title='Confusion matrix', xaxis_title='Predicted', yaxis_title='True', width=500, height=50*len(label_typemap.keys()))

    faileds = get_faileds(validations, top=15, past=15)
    failed_plots = []
    for name, pred in faileds:
        objrow = valid_table[valid_table["source_id"] == name].iloc[0]
        plot = __create_valid_plot__(objrow)
        plot["layout"]["title"]["text"] += "- pred. as {}".format(pred)
        failed_plots.append(plot)

    clear_output(wait=True)
    display(fig)
    print(faileds)
    display(confmatrix_plot)
    for plot in failed_plots:
        display(plot)

def get_faileds(validations, top=10, past=10):
    """
    Returns a list of indices of validation sets where the F1 score is less than 0.5
    """
    def get_mode(arr):
        bins = np.bincount(arr)
        return np.argmax(bins)
    
    names = validations[-1]["names"]

    considered = validations[-past:]
    true = np.array(validations[-1]["true_y"]).T
    pred_matrix = np.transpose(np.array([v["pred_y"] for v in considered]))
    print(pred_matrix)
    predictions = np.array([get_mode(pred_matrix[i]) for i in range(len(pred_matrix))])
    incorrects = predictions != true
    indices = np.where(incorrects)[0]

    return [(names[i], predictions[i]) for i in indices][-top:]


    

# def plot_faileds(names, valid, top=10):
#     data, label, types, vnames = valid
#     failed = [vnames.index(name) for name in names]
#     failed_data = data[failed]

#     for i in len(names):
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(x=failed_data[:, 2], y=failed_data[:, 0], mode='markers'))
