import networkx as nx
import numpy as np
from tensorflow.data import Dataset
from tensorflow import SparseTensor


def get_graph_from_file(filename):
    if filename is None:
        raise AssertionError("File name is None!")
    G = nx.read_edgelist(filename, comments="#", nodetype=int, data=(('weight', float),))
    return G


def get_data_loader(S, batch_size=128):
    training_features = S.astype('float32')
    training_dataset = Dataset.from_tensor_slices(training_features)
    training_dataset = training_dataset.batch(batch_size)
    # training_dataset = training_dataset.shuffle(training_features.shape[0])  # shuffle all node in graph. |V| = training_feature.shape[0] = A.todense().shape[0]
    training_dataset = training_dataset.prefetch(batch_size * 4)
    return training_dataset


def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return SparseTensor(indices, coo.data, coo.shape)

def get_tf_dataset(A, L):
    A_ = convert_sparse_matrix_to_sparse_tensor(A)
    L_ = convert_sparse_matrix_to_sparse_tensor(L)
    A_ds = Dataset.from_tensor_slices(A_)
    L_ds = Dataset.from_tensor_slices(L_)

    dataset = Dataset.zip((A_ds, L_ds))
    return dataset
