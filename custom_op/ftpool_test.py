import tensorflow as tf
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import keras
from user_ops import ft_pool


tf.enable_eager_execution()
print('tf is executing eagerly: ', tf.executing_eagerly())

input = np.ones([1,5,5,1], dtype=np.float)
for i in range(5):
    input[0,i,i,0] = 2
output = ft_pool(input, (1.4,1.4), (1.4,1.4))

print("input shape", input.shape, "output shape", output.shape)
print("input tensor is \n", input[0, ..., 0])
print("output tensor is \n", output[0, ..., 0])
'''
mnist = keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

model = keras.models.Sequential()
model.add(keras.layers.InputLayer(batch_input_shape=(1,28,28,1)))
model.add(keras.layers.Conv2D(1, (3,3), activation='relu', name='conv2D_1'))
model.add(keras.layers.Lambda(lambda x: ft_pool(x, (2.0,2.0), 10)))
#model.add(keras.layers.AveragePooling2D(pool_size=(5,5), strides=(2,2), padding='same'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(10, activation='softmax', name='dense_1'))
print('before compile')
print('compiling model', model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']))
print('after compile')
print(model.summary())
for i in range(3):
    print(model.train_on_batch(x_train[i:i+1, ...], y_train[i:i+1, ...]))
print(model.evaluate(x_test, y_test, batch_size=50))
'''
