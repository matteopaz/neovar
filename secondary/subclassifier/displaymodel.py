import torch
from autoencoder import Morphologic
from torchview import draw_graph

model = Morphologic(64, 7, features=2)

graph = draw_graph(model, input_size=(1, 64, 2), hide_inner_tensors=True, hide_module_functions=True, depth=1)

graph.visual_graph.render("morpho_model.png", format="png")