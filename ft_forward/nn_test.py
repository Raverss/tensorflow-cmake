from user_ops import ft_pool
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import math
import pickle
import json
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

import sys
sys.path.append(os.path.join('..', '..', 'keras_frac'))
#from fractional_maxpooling import FractionalPooling2D
#import resnet

init = "he_uniform"

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)

class max_pool_net:
    def __init__(self, opt, batch_input_shape, classes, k_size=(3, 3), filters=2, blocks=2):
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.InputLayer(batch_input_shape=batch_input_shape))
        self.model.add(
            keras.layers.Conv2D(filters, k_size, activation='r2c2elu', padding='same', kernel_initializer=init))
        self.model.add(keras.layers.Conv2D(filters, k_size, activation='relu', padding='same', kernel_initializer=init))
        self.model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        for i in range(2, blocks + 1):
            self.model.add(
                keras.layers.Conv2D(filters * i, k_size, activation='relu', padding='same', kernel_initializer=init))
            self.model.add(
                keras.layers.Conv2D(filters * i, k_size, activation='relu', padding='same', kernel_initializer=init))
            self.model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(classes, activation='softmax'))
        self.model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
        print(self.model.summary())

    def train(self, batch_size, epochs, datagen=None, train_data=None, callbacks=None):
        if datagen is None and train_data is None:
            print('neither data or generator was passed')
        elif datagen is None and train_data is not None:
            print('training on array data, network type:', type(self).__name__)
            self.history = self.model.fit(x=train_data[0], y=train_data[1], validation_data=train_data[2:],
                                          batch_size=batch_size, epochs=epochs, callbacks=callbacks)
        else:
            print('training on datagen data, network type:', type(self).__name__)
            self.history = self.model.fit_generator(datagen, validation_data=train_data[2:], batch_size=batch_size,
                                                    epochs=epochs, callbacks=callbacks)
        self.weights = self.model.get_weights()

    def restart_session(self):
        keras.backend.clear_session()

    def get_history(self):
        return self.history

    def get_weights(self):
        return self.weights

    def load_weights(self):
        self.model.load_weights(self.weights)


class ft_pool_net(max_pool_net):
    def __init__(self, opt, batch_input_shape, classes, k_size=(3, 3), filters=2, blocks=2):
        strides = (math.sqrt(2), math.sqrt(2))
        #strides = (2.0,2.0)
        pool_size = (math.sqrt(2) * 2, math.sqrt(2) * 2)
        #pool_size = (2.0,2.0)
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.InputLayer(batch_input_shape=batch_input_shape))
        self.model.add(keras.layers.Conv2D(filters, k_size, activation='relu', padding='same', kernel_initializer=init))
        self.model.add(keras.layers.Lambda(lambda x: ft_pool(x, strides, pool_size)))
        self.model.add(keras.layers.Conv2D(filters, k_size, activation='relu', padding='same', kernel_initializer=init))
        self.model.add(keras.layers.Lambda(lambda x: ft_pool(x, strides, pool_size)))
        for i in range(2, blocks+1):
            self.model.add(keras.layers.Conv2D(filters*i, k_size, activation='relu', padding='same', kernel_initializer=init))
            self.model.add(keras.layers.Lambda(lambda x: ft_pool(x, strides, pool_size)))
            self.model.add(keras.layers.Conv2D(filters*i, k_size, activation='relu', padding='same', kernel_initializer=init))
            self.model.add(keras.layers.Lambda(lambda x: ft_pool(x, strides, pool_size)))
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(classes, activation='softmax'))
        self.model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
        print(self.model.summary())

EPOCHS = 30
BATCH_SIZE = 50
iter = 1

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

ft_net_h = []
path = os.path.join('results', 'mine_with_div')

for _ in range(iter):
    ft_net = ft_pool_net(keras.optimizers.Adadelta(), (BATCH_SIZE,) + x_train.shape[1:], y_train.shape[1], blocks=2)
    ft_net.train(BATCH_SIZE, EPOCHS, train_data=(x_train, y_train, x_test, y_test), callbacks=[keras.callbacks.ReduceLROnPlateau(patience=5, factor=.5, verbose=1)])
    #ft_net_h.append([ft_net.get_history().history['accuracy'], ft_net.get_history().history['val_accuracy']])
    #ft_net.restart_session()
#with open(os.path.join(path,'ft_net_h.pkl'), 'wb') as f:
#    pickle.dump(ft_net_h, f)