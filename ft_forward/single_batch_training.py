from user_ops import ft_pool
import numpy as np
import tensorflow as tf
import keras
from keras.backend.tensorflow_backend import set_session
import cv2
import math
import random
import sys
import os

################################################################################
#                                   params                                     #
################################################################################
BATCHES = 2
PATCH_START = [20, 20]
in_shape = [1,6,6,3]
stride = [1.5,1.5]
pool = [3.0, 3.0]
config_path = "test_config.h5"
################################################################################
#                                   load data                                  #
################################################################################
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
y_train = keras.utils.to_categorical(y_train)
# get random image and normalize it
index = random.randint(0, x_train.shape[0]-2)
x = x_train[index:index+1, PATCH_START[0]:PATCH_START[0]+in_shape[1], PATCH_START[1]:PATCH_START[1]+in_shape[2], 0:in_shape[3]]
#y = [[10.0]]
y = y_train[index:index+1,:]
################################################################################
#                                   net def                                    #
################################################################################
def get_model():
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(batch_input_shape=in_shape))
    model.add(keras.layers.Conv2D(1, 3, activation='relu', padding='same', use_bias=False))
    model.add(keras.layers.Lambda(lambda x: ft_pool(x, stride, pool)))
    model.add(keras.layers.Conv2D(32, 3, activation='relu', padding='same', use_bias=False))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, activation='softmax', use_bias=False))
    #model.add(keras.layers.Dense(1, activation='linear', use_bias=False))
    model.compile('sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    #model.compile('sgd', loss='mse', metrics=['accuracy'])
    print(model.summary())
    return model
################################################################################
#                                   save weights                               #
################################################################################
model = get_model()
model.save_weights(config_path)
################################################################################
#                                   run it - CPU                               #
################################################################################
config = tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=4, allow_soft_placement=True, device_count = {'CPU' : 1, 'GPU' : 0})
session = tf.Session(config=config)
set_session(session)
model = get_model()
model.load_weights(config_path)
cpu_weights = model.get_weights()
for _ in range(BATCHES):
    cpu_history = model.train_on_batch(x,y)
################################################################################
#                                   run it - GPU                               #
################################################################################
config = tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=4, allow_soft_placement=True, device_count = {'CPU' : 1, 'GPU' : 1})
session = tf.Session(config=config)
set_session(session)
model = get_model()
model.load_weights(config_path)
gpu_weights = model.get_weights()
for _ in range(BATCHES):
    gpu_history = model.train_on_batch(x,y)
################################################################################
#                         cpu, gpu weights eq check                            #
################################################################################
for w_pair in zip(cpu_weights, gpu_weights):
    assert np.sum(np.abs(w_pair[0] - w_pair[1])) == 0.0, "Assert for cpu weights == gpu weights failed!"