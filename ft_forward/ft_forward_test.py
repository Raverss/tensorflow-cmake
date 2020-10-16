#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import keras
from user_ops import ft_pool

# Expected input
IN_M = np.asarray(
    [[0, 1, 2, 3, 4, 5],
     [1, 2, 3, 4, 5, 6],
     [2, 3, 4, 5, 6, 7],
     [3, 4, 5, 6, 7, 8],
     [4, 5, 6, 7, 8, 9],
     [5, 6, 7, 8, 9, 10]], dtype=np.float32)

# Expected output
EXP_M = np.asarray(
    [[0.6664624, 1.8332312, 3.3332314, 4.833231],
     [1.8332312, 3.,        4.5,       6.],
     [3.3332314, 4.5,       6.,        7.5],
     [4.833231,  6.,        7.5,       9.]], dtype=np.float32)

# Failed output
EXP_M_F = np.asarray(
    [[0.66, 1.1, 3.3, 4.3],
     [1.83, 3.,  4.5, 6.],
     [3.33, 4.5, 6.,  7.5],
     [4.3,  6.,  7.5, 9.]], dtype=np.float32)

STRIDE = [1.5, 1.5]
POOL_SIZE = [3.0, 3.0]

def matrix_diff(out_m, exp_m):
    """
        >>> matrix_diff(EXP_M, EXP_M)
        0.0
        >>> matrix_diff(EXP_M, EXP_M_F)
        1.8458494
    """
    diff = np.abs(out_m - exp_m)
    return np.sum(diff)

def nn_predict():
    img = np.expand_dims(IN_M, 0)
    img = np.expand_dims(img, 3)
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(batch_input_shape=img.shape))
    model.add(keras.layers.Lambda(lambda x: ft_pool(x, STRIDE, POOL_SIZE)))
    model.compile('adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
    return  model.predict(img)[0,...,0]


def nn_predict_test(exp):
    """
        >>> nn_predict_test(EXP_M)
        0.0
    """
    out = nn_predict()
    diff = matrix_diff(out, exp)
    return diff


if __name__ == '__main__':
    print("Please run as: python -m doctest -v ft_forward_test.py")
