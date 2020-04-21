from keras.layers import Lambda, Subtract
from keras import backend as KBack
from keras.optimizers import SGD, Adam

from static_graph_embedding import StaticGraphEmbedding
from utils import graph_util
from utils.sdne_utils import *
import networkx as nx
from time import time

from utils.visualize import plot_embeddings


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
        graph = nx.Graph(graph)
        if not graph and not edge_f:
            raise Exception('graph/edge_f needed')
        # if not graph:
        #     graph = graph_util.loadGraphFromEdgeListTxt(edge_f)

        S = nx.to_scipy_sparse_matrix(graph)
        t1 = time()
        S = (S + S.T) / 2  # TODO ???
        self._node_num = graph.number_of_nodes()

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
        x_diff1 = Subtract()([x_hat1, x1])
        x_diff2 = Subtract()([x_hat2, x2])
        y_diff = Subtract()([y2, y1])

        # Objectives
        def weighted_mse_x(y_true, y_pred):
            return KBack.sum(KBack.square(y_pred * y_true[:, 0:self._node_num]), axis=-1) \
                   / y_true[:, self._node_num]

        def weighted_mse_y(y_true, y_pred):
            min_batch_size = KBack.shape(y_true)[0]
            return KBack.reshape(KBack.sum(KBack.square(y_pred), axis=-1), [min_batch_size, 1]) \
                   * y_true

        # Model
        self._model = Model(input=x_in, output=[x_diff1, x_diff2, y_diff])
        sgd = SGD(lr=self._xeta, decay=1e-5, momentum=0.99, nesterov=True)

        # adam = Adam(lr=self._xeta, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self._model.compile(
            optimizer=sgd,
            loss=[weighted_mse_x, weighted_mse_x, weighted_mse_y],
            loss_weights=[1, 1, self._alpha]
        )

        self._model.fit_generator(
            generator=batch_generator_sdne(S, self._beta, self._n_batch, shuffle=True),
            epochs=self._num_iter,
            steps_per_epoch=S.nonzero()[0].shape[0] // self._n_batch,
            verbose=1
        )

        # Get embedding for all points
        self._Y = model_batch_predictor(self._autoencoder, S, self._n_batch)
        t2 = time()
        self.save_autoencoder()
        return self._Y, (t2 - t1)

    def save_autoencoder(self):
        # Save the autoencoder and its weights
        if self._weightfile is not None:
            save_weights(self._encoder, self._weightfile[0])
            save_weights(self._decoder, self._weightfile[1])

        if self._modelfile is not None:
            save_model(self._encoder, self._modelfile[0])
            save_model(self._decoder, self._modelfule[1])

        if self._savefilesuffix is not None:
            save_weights(self._encoder, 'encoder_weights_' + self._savefilesuffix + '.hdf5')
            save_weights(self._decoder, 'decoder_weights_' + self._savefilesuffix + '.hdf5')
            save_model(self._encoder, 'encoder_model_' + self._savefilesuffix + '.json')
            save_model(self._decoder, 'decoder_model_' + self._savefilesuffix + '.json')
            # Save the embedding
            np.savetxt('embedding_' + self._savefilesuffix + '.txt', self._Y)

    def get_embedding(self, filesuffix=None):
        return self._Y if filesuffix is None else np.loadtxt('embedding_' + filesuffix + '.txt')

    def get_edge_weight(self, i, j, embed=None, filesuffix=None):
        if embed is None:
            if filesuffix is None:
                embed = self._Y
            else:
                embed = np.loadtxt('embedding_' + filesuffix + '.txt')
        if i == j:
            return 0
        else:
            S_hat = self.get_reconst_from_embed(embed[(i, j), :], filesuffix)
            return (S_hat[i, j] + S_hat[i, j]) / 2

    def get_reconst_from_embed(self, embed, node_l=None, filesuffix=None):
        decoder = None
        if filesuffix is None:
            decoder = self._decoder
        else:
            try:
                decoder = model_from_json(open('decoder_model_' + filesuffix + '.json').read())
            except:
                print("Error reading file: {0}. Cannot load previous model".format(
                    'decoder_model' + filesuffix + '.hdf5'))
                exit()

        if node_l is not None:
            return decoder.predict(embed, batch_size=self._n_batch)[:, node_l]
        else:
            return decoder.predict(embed, batch_size=self._n_batch)

    def get_reconstructed_adj(self, embed=None, node_l=None, filesuffix=None):
        if embed is None:
            if filesuffix is None:
                embed = self._Y
            else:
                embed = np.loadtxt('embedding_' + filesuffix + '.txt')
        S_hat = self.get_reconst_from_embed(embed, node_l, filesuffix)
        return graphify(S_hat)


if __name__ == '__main__':
    G = nx.karate_club_graph()
    G = G.to_directed()
    res_pre = 'results/testKarate'
    graph_util.print_graph_stats(G)

    t1 = time()
    embedding = SDNE(d=2, beta=5, alpha=1e-5, nu1=1e-6, nu2=1e-6, K=3,
                     n_units=[50, 15], rho=0.3, n_iter=50, xeta=0.01, n_batch=50,
                     modelfile=None,
                     weightfile=None)

    embedding.learn_embedding(graph=G, edge_f=None, is_weighted=True, no_python=True)
    print("SDNE: \n\tTraining time: %f" % (time() - t1))
    plot_embeddings(G, embedding.get_embedding())
