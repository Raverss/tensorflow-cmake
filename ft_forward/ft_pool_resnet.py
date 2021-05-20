import json
import math
import os
import random
import sys
import abc
from user_ops import ft_pool

import tensorflow as tf
import keras
import keras.backend as K
import numpy as np
from keras.callbacks import Callback, EarlyStopping, LambdaCallback, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.datasets import cifar10
from keras.backend.tensorflow_backend import set_session
from keras.layers import (
    Activation,
    AveragePooling2D,
    BatchNormalization,
    Conv2D,
    Convolution2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
)

from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.utils.np_utils import to_categorical
from tensorflow.python import debug as tf_debug

# Model parameter
# ----------------------------------------------------------------------------
#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
# ----------------------------------------------------------------------------
# ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
# ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
# ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
# ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
# ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# ---------------------------------------------------------------------------


class resnet_base(abc.ABC):
    def __init__(self, n=18):
        self.batch_size = 50  # orig paper trained all networks with batch_size=128
        self.data_augmentation = True
        self.num_classes = 10
        self.subtract_pixel_mean = True
        # Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
        self.version = 1
        self.test_acc = []
        # Computed depth from supplied model parameter n
        if self.version == 1:
            self.depth = n * 6 + 2
        elif self.version == 2:
            self.depth = n * 9 + 2
        # Load the CIFAR10 data.
        (self.x_train, self.y_train), (self.x_test,
                                       self.y_test) = cifar10.load_data()
        # Input image dimensions.
        self.input_shape = self.x_train.shape[1:]
        self.preprocess()

    def preprocess(self):
        # Normalize data.
        self.x_train = self.x_train.astype("float32") / 255
        self.x_test = self.x_test.astype("float32") / 255
        # If subtract pixel mean is enabled
        if self.subtract_pixel_mean:
            x_train_mean = np.mean(self.x_train, axis=0)
            self.x_train -= x_train_mean
            self.x_test -= x_train_mean

        print("x_train shape:", self.x_train.shape)
        print(self.x_train.shape[0], "train samples")
        print(self.x_test.shape[0], "test samples")
        print("y_train shape:", self.y_train.shape)
        # Convert class vectors to binary class matrices.
        self.y_train = keras.utils.to_categorical(
            self.y_train, self.num_classes)
        self.y_test = keras.utils.to_categorical(self.y_test, self.num_classes)

    def __lr_schedule(self, epoch):
        lr = 1e-3
        if epoch > 180:
            lr *= 0.5e-3
        elif epoch > 160:
            lr *= 1e-3
        elif epoch > 120:
            lr *= 1e-2
        elif epoch > 80:
            lr *= 1e-1
        return lr

    def __conv_bn_order(self, x, conv, conv_first, activation, batch_normalization):
        if conv_first:
            x = conv(x)
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
        else:
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
            x = conv(x)
        return x

    @abc.abstractmethod
    def resnet_layer(
        self,
        inputs,
        trainable=True,
        in_channels=None,
        num_filters=16,
        kernel_size=3,
        strides=1,
        activation="relu",
        batch_normalization=True,
        conv_first=True,
    ):
        """ Implement me! """
        pass

    def resnet_v1(self):
        if (self.depth - 2) % 6 != 0:
            raise ValueError("depth should be 6n+2 (eg 20, 32, 44 in [a])")
        # Start model definition.
        num_filters = 16
        num_res_blocks = int((self.depth - 2) / 6)

        inputs = Input(shape=self.input_shape)
        prev_filters = 16
        x = self.resnet_layer(inputs=inputs, num_filters=16,
                              in_channels=3, strides=1)
        # Instantiate the stack of residual units
        for stack in range(3):
            for res_block in range(num_res_blocks):
                strides = 1
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    strides = 2  # downsample
                y = self.resnet_layer(
                    inputs=x, num_filters=num_filters, in_channels=prev_filters, strides=strides)
                y = self.resnet_layer(
                    inputs=y, num_filters=num_filters, in_channels=num_filters, activation=None)
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = self.resnet_layer(
                        inputs=x,
                        num_filters=num_filters,
                        kernel_size=1,
                        strides=strides,
                        activation=None,
                        batch_normalization=False,
                    )
                x = keras.layers.add([x, y])
                x = Activation("relu")(x)
                prev_filters = num_filters
            num_filters *= 2

        x = AveragePooling2D(pool_size=8)(x)
        y = Flatten()(x)
        outputs = Dense(self.num_classes, activation="softmax",
                        kernel_initializer="he_normal")(y)
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def build_network(self):
        if self.version == 2:
            # model = resnet_v2(input_shape=input_shape, depth=depth)
            raise NotImplementedError
        else:
            self.model = self.resnet_v1()
        self.model.compile(
            loss="categorical_crossentropy", optimizer=Adam(learning_rate=self.__lr_schedule(0), clipvalue=0.1), metrics=["accuracy"]
        )
        lr_scheduler = LearningRateScheduler(self.__lr_schedule)
        chckp = ModelCheckpoint(
            "saved_models/" + type(self).__name__ + ".h5", monitor="val_accuracy", save_best_only=True, save_weights_only=True
        )
        self.callbacks = [lr_scheduler]

    def train(self, epochs=200, verbose=1):
        if not self.data_augmentation:
            print("Not using data augmentation.")
            self.history = self.model.fit(
                self.x_train,
                self.y_train,
                batch_size=self.batch_size,
                epochs=epochs,
                validation_data=(self.x_test, self.y_test),
                shuffle=True,
                callbacks=self.callbacks,
                verbose=verbose,
            )
        else:
            print("Using real-time data augmentation.")
            datagen = ImageDataGenerator(
                width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
            datagen.fit(self.x_train)
            self.history = self.model.fit_generator(
                datagen.flow(self.x_train, self.y_train,
                             batch_size=self.batch_size, shuffle=True),
                validation_data=(self.x_test, self.y_test),
                epochs=epochs,
                verbose=verbose,
                workers=4,
                callbacks=self.callbacks,
            )
        '''
        self.model.load_weights("saved_models/" + type(self).__name__ + ".h5")
        
        scores = self.model.evaluate(self.x_test, self.y_test, verbose=1)
        if "test_acc" in self.history.history.keys():
            self.history.history["test_acc"].append([scores[1]])
        else:
            self.history.history["test_acc"] = [scores[1]]
        self.history.history["test_acc"]
        print("Test loss:", scores[0])
        print("Test accuracy:", scores[1])
        '''

    def clear_session(self):
        keras.backend.clear_session()

    def save_history(self, folder):
        name = os.path.join(folder, type(self).__name__) + "{}.json".format(self.depth)
        if os.path.exists(name) and os.path.getsize(name) > 0:
            with open(name, "r") as f:
                data = json.load(f)
                for key in self.history.history.keys():
                    if key in data.keys():
                        data[key].append(
                            list(map(float, self.history.history[key])))
                    else:
                        data[key] = [
                            list(map(float, self.history.history[key]))]
            with open(name, "w") as f:
                json.dump(data, f)
        else:
            data = {}
            for key in self.history.history.keys():
                data[key] = [list(map(float, self.history.history[key]))]
            with open(name, "w") as f:
                json.dump(data, f)


class resnet(resnet_base):
    def resnet_layer(
        self,
        inputs,
        trainable=True,
        in_channels=None,
        num_filters=16,
        kernel_size=3,
        strides=1,
        activation="relu",
        batch_normalization=True,
        conv_first=True,
    ):
        conv = Conv2D(
            num_filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=l2(1e-4),
        )
        return self._resnet_base__conv_bn_order(inputs, conv, conv_first, activation, batch_normalization)


class FtPoolingResnet(resnet):
    def resnet_v1(self):
        print('build build')
        if (self.depth - 2) % 6 != 0:
            raise ValueError("depth should be 6n+2 (eg 20, 32, 44 in [a])")
        # Start model definition.
        num_filters = 16
        num_res_blocks = int((self.depth - 2) / 6)

        inputs = Input(shape=self.input_shape)
        prev_filters = 16
        x = self.resnet_layer(inputs=inputs, num_filters=16,
                              in_channels=3, strides=1)
        # Instantiate the stack of residual units
        for stack in range(3):
            for res_block in range(num_res_blocks):
                strides = 1
                if stack > 0 and res_block % math.ceil(num_res_blocks / 2) == 0:
                    pool_strides = (math.sqrt(2), math.sqrt(2))
                    pool_size = (math.sqrt(2) * 2, math.sqrt(2) * 2)
                    p = keras.layers.Lambda(lambda x: ft_pool(x, pool_strides, pool_size))(x)
                    y = self.resnet_layer(
                        inputs=p, num_filters=num_filters, in_channels=prev_filters, strides=strides)
                else:
                    y = self.resnet_layer(
                        inputs=x, num_filters=num_filters, in_channels=prev_filters, strides=strides)

                y = self.resnet_layer(
                    inputs=y, num_filters=num_filters, in_channels=num_filters, activation=None)

                if stack > 0 and res_block % math.ceil(num_res_blocks / 2) == 0:
                    # first layer but not first stack
                    # linear projection residual shortcut connection to match
                    # changed dims
                    pool_strides = (math.sqrt(2), math.sqrt(2))
                    pool_size = (math.sqrt(2) * 2, math.sqrt(2) * 2)
                    x = keras.layers.Lambda(lambda x: ft_pool(x, pool_strides, pool_size))(x)
                    x = self.resnet_layer(
                        inputs=x,
                        num_filters=num_filters,
                        kernel_size=1,
                        strides=strides,
                        activation=None,
                        batch_normalization=False,
                    )
                x = keras.layers.add([x, y])
                x = Activation("relu")(x)
                prev_filters = num_filters
            num_filters *= 2

        x = AveragePooling2D(pool_size=8)(x)
        y = Flatten()(x)
        outputs = Dense(self.num_classes, activation="softmax",
                        kernel_initializer="he_normal")(y)
        model = Model(inputs=inputs, outputs=outputs)
        return model

#config = tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=4, allow_soft_placement=True, device_count = {'CPU' : 1, 'GPU' : 0})
#session = tf.Session(config=config)
# set_session(session)


ftpoolresnet = FtPoolingResnet(n=18)
ftpoolresnet.build_network()
ftpoolresnet.train()
ftpoolresnet.save_history('')
#rt = resnet(n=18)
# rt.build_network()
# rt.train()
# rt.save_history('')
