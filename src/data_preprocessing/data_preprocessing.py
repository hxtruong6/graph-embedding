import networkx as nx
import numpy as np
import tensorflow as tf
from tensorflow import SparseTensor


def get_graph_from_file(filename):
    if filename is None:
        raise AssertionError("File name is None!")
    G = nx.read_edgelist(filename, comments="#", nodetype=int, data=(('weight', float),))
    return G


def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return SparseTensor(indices, coo.data, coo.shape)


def next_datasets(A, L, batch_size):
    '''

    :param A:
    :param L:
    :param batch_size:
    :return:
    '''
    dataset_size = A.shape[0]
    steps_per_epoch = (dataset_size - 1) // batch_size + 1
    i = 0
    while i < steps_per_epoch:
        index = np.arange(
            i * batch_size, min((i + 1) * batch_size, dataset_size))
        A_train = A[index, :].todense()
        L_train = L[index][:, index].todense()
        batch_inp = [A_train, L_train]

        yield i, batch_inp
        i += 1
