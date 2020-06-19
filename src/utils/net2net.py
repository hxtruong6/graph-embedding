import numpy as np


def net2wider(weights1, bias1, weights2, new_width):
    pass


def net2deeper(weights):
    '''

    :param weight: numpy array has shape(inp_size, out_size). input_size and out_size are number of units from source
                    to destination layer
    :return:
    '''
    _, out = weights.shape
    # new_weights = np.ndarray(np.eye(out))
    # new_weight = np.eye(out)

    new_weights = np.array(np.eye(out))
    new_bias = np.zeros((out, ))
    return new_weights, new_bias
