import tensorflow as tf
import networkx as nx
import numpy as np
import scipy.sparse as sparse

from data_preprocessing.data_preprocessing import get_tf_dataset
from utils.autoencoder import Autoencoder
from utils.graph_util import preprocess_graph


class StaticGE(object):
    def __init__(self, G, embedding_dim, hidden_dims, alpha=0.01, beta=2, nu1=0.001, nu2=0.001):
        super(StaticGE, self).__init__()
        self.G = nx.Graph(G)
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.alpha = alpha
        self.beta = beta

        self.input_dim = self.G.number_of_nodes()

        self.model = Autoencoder(
            input_dim=self.input_dim,
            embedding_dim=self.embedding_dim,
            hidden_dims=self.hidden_dims,
            v1=v1,
            v2=v2
        )

        self.A, self.L = self.create_A_L_matrix()
        self.tf_dataset = get_tf_dataset(self.A, self.L)

    def loss_calculate(self, X, L, alpha=0.01, beta=2):
        '''

        :param X:
        :param L:
        :param alpha:
        :param beta:
        :return:
        '''
        Y, X_hat = self.model(X)

        def loss_1st(Y, L):
            # D = tf.linalg.diag(tf.math.count_nonzero(S, axis=1))
            # D = tf.cast(D, tf.float32)
            # L = tf.math.subtract(D, S)  # L = D - A : laplacian eigenmaps

            # 2 * tr(Y^T * L * Y)
            return 2 * tf.linalg.trace(
                tf.linalg.matmul(tf.linalg.matmul(tf.transpose(Y), L), Y)
            )

        def loss_2nd(X_hat, X, beta):
            B = X * (beta - 1) + 1
            # print("B: ", B)
            # print("B shape: ", B.shape)
            return tf.reduce_sum(tf.pow((X_hat - X) * B, 2))

        loss_1 = loss_1st(Y, L)
        # print("1st: ", loss_1)
        loss_2 = loss_2nd(X_hat, X, beta)
        # print("2nd: ",loss_2)
        return loss_2 + alpha * loss_1

    def train(self, epochs=10, learning_rate=0.003):
        def train_func(loss, model, opt, X, L, alpha, beta):
            with tf.GradientTape() as tape:
                gradients = tape.gradient(
                    loss(model, X, L, alpha, beta),
                    model.trainable_variables
                )
            gradient_variables = zip(gradients, model.trainable_variables)
            opt.apply_gradients(gradient_variables)

        # ---------
        writer = tf.summary.create_file_writer('tmp')

        graph_embedding_list = []
        losses = []

        # tf.keras.backend.clear_session()
        opt = tf.optimizers.Adam(learning_rate=learning_rate)
        with writer.as_default():
            with tf.summary.record_if(True):
                for epoch in range(epochs):
                    epoch_loss = []
                    for step, batch in enumerate(self.tf_dataset):
                        A, L = batch
                        print(f"A= {A}")
                        print(f"L= {L}")
                #         A = A.todense()
                #         L = L.todense()
                #
                #     train_func(self.loss_calculate, self.model, opt, batch_features, L, alpha=self.alpha,
                #                beta=self.beta)
                #     # loss_values = self.loss_calculate(self.model, batch_features, S, beta_loss)
                #     # epoch_loss.append(loss_values)
                #     # tf.summary.scalar('loss', loss_values, step=epoch)
                #
                #     # embedding = autoencoder.get_embedding(S)
                #     # print(embedding.shape)
                #     if epoch % 400 == 0:
                #         # plot_embeded(embedding[:500, :])
                #         # graph_embedding_list.append(embedding)
                #         mean_epoch_loss = np.mean(epoch_loss)
                #         print(f"\tEpoch {epoch}: Loss = {mean_epoch_loss}")
                #         losses.append(mean_epoch_loss)
                #
                # # plot_loss(losses)
                # print(f"Loss = {losses[-1]}")

    def create_A_L_matrix(self):
        A = nx.to_scipy_sparse_matrix(self.G, format='csr')
        D = sparse.diags(A.sum(axis=1).flatten().tolist()[0])
        L = D - A
        return A, L

    def get_embedding(self, inputs):
        if inputs is None:
            inputs = nx.adj_matrix(self.G).todense()

        return self.model.get_embedding(inputs)


if __name__ == "__main__":
    S = np.array([
        [0, 2, 0, 4, 5],
        [2, 0, 1, 0, 6],
        [0, 1, 0, 0, 0],
        [4, 0, 0, 0, 0],
        [5, 6, 0, 0, 0]
    ])
    G = nx.from_numpy_matrix(S, create_using=nx.Graph)
    ge = StaticGE(G=G, embedding_dim=2, hidden_dims=[128, 512])
    ge.train()
