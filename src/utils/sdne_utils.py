import numpy as np

from keras.layers import Input, Dense
from keras.models import Model, model_from_json
import keras.regularizers as Reg


def get_encoder(node_num, d, K, n_units, nu1, nu2, activation_fn):
    '''

    :param node_num:
    :param d: dimension of graph embedding
    :param K: number layer Input (layer 0) -> Embedding layer (layer K)
    :param n_units: number unit of each layer
    :param nu1: L1
    :param nu2: L2
    :param activation_fn: type of activation
    :return:
    '''
    # Input
    x = Input(shape=(node_num,))

    # Encoder layers
    y = [None] * (K + 1)
    y[0] = x  # y[0] is assigned the input
    for i in range(K - 1):
        y[i + 1] = Dense(n_units[i], activation=activation_fn,
                         kernel_regularizer=Reg.l1_l2(l1=nu1, l2=nu2))(y[i])
    y[K] = Dense(d, activation=activation_fn,
                 kernel_regularizer=Reg.l1_l2(l1=nu1, l2=nu2))(y[K - 1])

    # Encoder model
    encoder = Model(input=x, output=y[K])
    return encoder


def get_decoder(node_num, d, K, n_units, nu1, nu2, activation_fn):
    '''

    :param node_num:
    :param d:
    :param K:
    :param n_units:
    :param nu1:
    :param nu2:
    :param activation_fn:
    :return:
    '''
    # Input
    y = Input(shape=(d,))

    # Decoder layers
    y_hat = [None] * (K + 1)
    y_hat[K] = y
    for i in range(K - 1, 0, -1):
        y_hat[i] = Dense(n_units[i - 1], activation=activation_fn,
                         kernel_regularizer=Reg.l1_l2(l1=nu1, l2=nu2))(y_hat[i + 1])
    y_hat[0] = Dense(node_num, activation=activation_fn,
                     kernel_regularizer=Reg.l1_l2(l1=nu1, l2=nu2))(y_hat[1])

    # Output
    x_hat = y_hat[0]  # decoder's output is also the actual output

    # Decoder Model
    decoder = Model(input=y, output=x_hat)
    return decoder


def get_autoencoder(encoder, decoder):
    '''

    :param encoder:
    :param decoder:
    :return:
    '''
    # Input
    x = Input(shape=(encoder.layers[0].input_shape[1],))
    # Generate embedding
    y = encoder(x)
    # Generate reconstruction
    x_hat = decoder(y)
    # Autoencoder Model
    autoecoder = Model(input=x, output=[x_hat, y])
    return autoecoder
