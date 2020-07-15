from os.path import join
import tensorflow as tf
from static_ge import StaticGE
from utils.autoencoder import Autoencoder
import networkx as nx
from math import ceil

from utils.visualize import plot_embedding
import numpy as np
import pandas as pd


def get_hidden_layer(prop_size, input_dim, embedding_dim):
    hidden_dims = [input_dim]
    while ceil(hidden_dims[-1] * prop_size) > embedding_dim:
        hidden_dims.append(ceil(hidden_dims[-1] * prop_size))

    del hidden_dims[0]
    return hidden_dims


def handle_expand_model(model: Autoencoder, input_dim, net2net_applied=False, prop_size=0.3):
    if input_dim == model.get_input_dim():
        return model

    # NOTE: suppose just for addition nodes to graph
    model.expand_first_layer(layer_dim=input_dim)

    if net2net_applied is False:
        return model

    layers_size = model.get_layers_size()
    index = 0
    while index < len(layers_size) - 1:
        layer_1_dim, layer_2_dim = layers_size[index]
        suitable_dim = ceil(layer_1_dim * prop_size)
        if suitable_dim > layer_2_dim:
            # the prev layer before embedding layer
            if index == len(layers_size) - 2:
                model.deeper(pos_layer=index)
                # model.info()
            else:
                added_size = suitable_dim - layer_2_dim
                model.wider(added_size=added_size, pos_layer=index)
                index += 1
        else:
            index += 1
        layers_size = model.get_layers_size()
    return model


def save_weights_model(weights, filepath):
    pd.DataFrame(weights).to_json(filepath, orient='split')


def load_weights_model(filepath):
    weights = pd.read_json(filepath, orient='split').to_numpy()
    for layer_index in range(len(weights[0])):
        weights[0][layer_index][0] = np.array(weights[0][layer_index][0], dtype=np.float32)
        weights[0][layer_index][1] = np.array(weights[0][layer_index][1], dtype=np.float32)
        weights[1][layer_index][0] = np.array(weights[1][layer_index][0], dtype=np.float32)
        weights[1][layer_index][1] = np.array(weights[1][layer_index][1], dtype=np.float32)

    return weights


def get_hidden_dims(layers_size):
    hidden_dims = []
    for i, (l1, l2) in enumerate(layers_size):
        if i == 0:
            continue
        hidden_dims.append(l1)
    return hidden_dims


class DynGE(object):
    def __init__(self, graphs, embedding_dim, init_hidden_dims=None, v1=0.001, v2=0.001):
        super(DynGE, self).__init__()
        if init_hidden_dims is None:
            init_hidden_dims = []
        self.graphs = graphs
        self.graph_len = len(graphs)
        self.embedding_dim = embedding_dim
        self.init_hidden_dims = init_hidden_dims
        self.v1 = v1
        self.v2 = v2
        self.static_ges = []
        self.model_weight_paths = []
        # self.models = []

    def get_all_embeddings(self):
        return [ge.get_embedding() for ge in self.static_ges]

    def get_embedding(self, index):
        if index < 0 or index >= self.graph_len:
            raise ValueError("index is invalid!")
        return self.static_ges[index].get_embedding()

    def train(self, prop_size=0.4, batch_size=64, epochs=100, filepath="../models/", skip_print=5,
              net2net_applied=False):
        init_hidden_dims = get_hidden_layer(prop_size=prop_size, input_dim=len(self.graphs[0].nodes()),
                                            embedding_dim=self.embedding_dim)
        # print(init_hidden_dims)
        model = Autoencoder(
            input_dim=len(self.graphs[0].nodes()),
            embedding_dim=self.embedding_dim,
            hidden_dims=init_hidden_dims,
            v1=self.v1,
            v2=self.v2
        )
        ge = StaticGE(G=self.graphs[0], model=model)
        ge.train(batch_size=batch_size, epochs=epochs, skip_print=skip_print)

        self.static_ges.append(ge)
        self.model_weight_paths.append(join(filepath, "graph_0.json"))
        save_weights_model(weights=ge.get_model().get_weights(), filepath=self.model_weight_paths[0])
        # print("Model_0: ", model.get_weights())
        print(model.get_layers_size())

        for i in range(1, len(self.graphs)):
            graph = nx.Graph(self.graphs[i])
            input_dim = len(graph.nodes())
            prev_model = self._create_prev_model(index=i)
            # print(f"Model_{i}:{prev_model.get_weights()}")
            # prev_model.info()
            curr_model = handle_expand_model(model=prev_model, input_dim=input_dim,
                                             prop_size=prop_size, net2net_applied=net2net_applied)
            print(curr_model.get_layers_size())
            ge = StaticGE(G=graph, model=curr_model)

            ge.train(batch_size=batch_size, epochs=epochs, skip_print=skip_print)

            self.static_ges.append(ge)
            self.model_weight_paths.append(join(filepath, f"graph_{i}.json"))
            save_weights_model(weights=ge.get_model().get_weights(), filepath=self.model_weight_paths[i])

    def _create_prev_model(self, index):
        prev_ge = self.static_ges[index - 1]
        prev_hidden_dims = get_hidden_dims(layers_size=prev_ge.get_model().get_layers_size())
        prev_input_dim = len(self.graphs[index - 1].nodes())
        model = Autoencoder(
            input_dim=prev_input_dim,
            embedding_dim=self.embedding_dim,
            hidden_dims=prev_hidden_dims,
            v1=self.v1,
            v2=self.v2
        )

        prev_weights = load_weights_model(self.model_weight_paths[index - 1])
        model.set_weight(weights=prev_weights)
        return model


if __name__ == "__main__":
    g1 = nx.complete_graph(100)
    g2 = nx.complete_graph(220)
    graphs = [g1, g2]
    dy_ge = DynGE(graphs=graphs, embedding_dim=2)
    dy_ge.train(prop_size=0.5, epochs=100, skip_print=20, net2net_applied=False)
    embeddings = dy_ge.get_all_embeddings()
    for e in embeddings:
        plot_embedding(embedding=e)
