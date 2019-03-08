import tensorflow as tf
import numpy as np
from user_ops import matrix_add

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

input_ = tf.keras.layers.Input(shape=(28, 28, 1))
x = tf.keras.layers.Conv2D(2, (3,3))(input_)
#x = tf.keras.layers.Lambda(lambda x: tf.nn.max_pool(x, (1,2,2,1), (1,2,2,1), "SAME"))(x)
x = tf.keras.layers.Lambda(lambda x: matrix_add(x, (2,2)))(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(10, activation=tf.nn.relu)(x)
x = tf.keras.layers.Dropout(0.2)(x)
output_ = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(x)

model = tf.keras.Model(inputs=input_, outputs=output_)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
print(model.evaluate(x_test, y_test))
