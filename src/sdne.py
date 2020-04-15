from keras.layers import Lambda, merge

from static_graph_embedding import StaticGraphEmbedding
from utils.sdne_utils import *
import networkx as nx
from time import time


class SDNE(StaticGraphEmbedding):
    def __init__(self, *hyper_dict, **kwargs):
        '''
        Initialize the SDNE class
        :param hyper_dict:
        :param kwargs:
        '''

        hyper_params = {
            'method_name': 'sdne',
            'actfn': 'relu',
            'modelfile': None,
            'weightfile': None,
            'savefilesuffix': None
        }
        hyper_params.update(kwargs)

        for key in hyper_params.keys():
            self.__setattr__('_%s' % key, hyper_params[key])

        for dictionary in hyper_dict:
            for key in dictionary:
                self.__setattr__('_s%' % key, dictionary[key])

    def get_method_name(self):
        return self._medthod_name

    def get_method_summary(self):
        return '%s_%d' % (self._method_name, self._d)

    def learn_embedding(self, graph=None, edge_f=None,
                        is_weighted=False, no_python=False):
        if not graph and not edge_f:
            raise Exception('graph/edge_f needed')
        # if not graph:
        #     graph = graph_util.loadGraphFromEdgeListTxt(edge_f)

        S = nx.to_spicy_sparse_matrix(graph)
        t1 = time()
        S = (S + S.T) / 2  # TODO ???
        self._node_num = graph.number_of_node()

        # Generate encoder, decoder and autoencoder
        self._num_iter = self._n_iter

        # If cannot use previous step information, initalize new models
        self._encoder = get_encoder(self._node_num, self._d, self._K, self._n_units,
                                    self._nu1, self._nu2, self._actfn)

        self._decoder = get_decoder(self._node_num, self._d, self._K, self._n_units,
                                    self._nu1, self._nu2, self._actfn)

        self._autoencoder = get_autoencoder(self._encoder, self._decoder)

        # Initialize self._model
        # Input TODO: ???
        x_in = Input(shape=(2 * self._node_num,), name='x_in')
        x1 = Lambda(lambda x: x[:, 0:self._node_num],
                    output_shape=(self._node_num,))(x_in)
        x2 = Lambda(lambda x: x[:, self._node_num:2 * self._node_num],
                    output_shape=(self._node_num,))(x_in)

        # Process inputs
        [x_hat1, y1] = self._autoencoder(x1)
        [x_hat2, y2] = self._autoencoder(x2)

        # Output
        x_diff1 = merge([x_hat1, x1])
