import plotly.graph_objects as go
from IPython.display import clear_output, display
import sklearn.metrics as metrics
import numpy as np


full_typemap = {
    "null": 0,
    "nova": 1,
    "sn": 2,
    "yso": 3,
    "agn": 4,
    "cep": 5,
    "rvt": 5,
    "rr": 6,
    "mira": 7,
    "sr": 8,
    "ea": 9,
    "ew": 10
}

label_typemap ={
    "null": 0,
    "nova": 1,
    "sn": 1,
    "yso": 3,
    "agn": 2,
    "cep": 2,
    "rr": 2,
    "mira": 2,
    "sr": 2,
    "ea": 2,
    "ew": 2
}

n_types = len(set(full_typemap.values()))
n_classes = len(set(label_typemap.values()))

def training_plot(trainlosses, validation_pairs):
    """
    Plots and displays a plotly graph of the training and validation losses, along with a confusion matrix.
    Trainlosses is a list of cross entropy losses for each epoch.
    validation_pairs is a list of pairs of lists, where the first list is the true string types and the second list is the predicted types out of the model
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=list(range(len(trainlosses))), y=trainlosses, mode='lines', name='Train loss'))
    
    f1_scores = [metrics.f1_score([label_typemap[t] for t in true_types], pred, average='weighted') for true_types, pred in validation_pairs]

    fig.add_trace(go.Scatter(x=list(range(len(f1_scores))), y=f1_scores, mode='lines', name='Validation F1 score'))
    fig.update_layout(title='Training and validation loss', xaxis_title='Epoch', yaxis_title='Loss')

    conf_matrix = np.zeros((n_classes, n_types))
    true_types, pred_labels = validation_pairs[-1]
    for true, pred in zip(true_types, pred_labels):
        conf_matrix[pred, full_typemap[true]] += 1
    
    confmatrix_plot = go.Figure(go.Heatmap(z=conf_matrix, x=[k for k in full_typemap.keys()], y=[k for k in label_typemap.keys()], colorscale='Viridis'))

    clear_output(wait=True)
    display(fig)
    display(confmatrix_plot)
