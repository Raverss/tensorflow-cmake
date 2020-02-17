import tensorflow as tf
import numpy as np
import os
from user_ops import ft_pool

#os.environ['CUDA_VISIBLE_DEVICES'] = ''

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
print(x_train.dtype)

f = open('accuracy', 'w')
iterations = 1
abc = 5.0

acc = [0,0]
for iteration in range(0, iterations):
    print("... starting {}-th iteration - ft pooling ...".format(iteration))
    input_ = tf.keras.layers.Input(shape=(28, 28, 1), name='input_layer')
    x = tf.keras.layers.Conv2D(2, (3,3), activation=tf.nn.relu, name='conv2D_1')(input_)
    x = tf.keras.layers.Lambda(lambda x: ft_pool(x, (1.4,1.4), abc), name='lambda')(x)
    x = tf.keras.layers.Conv2D(2, (3,3), activation=tf.nn.relu, name='conv2D_2')(x)
    x = tf.keras.layers.Flatten(name='flatten')(x)
    x = tf.keras.layers.Dense(10, activation=tf.nn.relu, name='dense_1')(x)
    x = tf.keras.layers.Dropout(0.2, name='dropout')(x)
    output_ = tf.keras.layers.Dense(10, activation=tf.nn.softmax, name='dense_2')(x)

    abc += 20.0

    model = tf.keras.Model(inputs=input_, outputs=output_)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())
    #model.train_on_batch(x_train[:4, ...], y_train[:4, ...])
    hist = model.fit(x_train, y_train, epochs=2, batch_size=1)
    acc[0] += hist.history['acc'][0]
    acc[1] += hist.history['acc'][1]
    tf.keras.backend.clear_session()
print('overall accuracy is {} for first epoch and {} for second epoch'.format(
    acc[0]/iterations, acc[1]/iterations), file=f, end='\n')

acc = [0,0]
for iteration in range(0, iterations):
    print("... starting {}-th iteration ...".format(iteration))
    input_ = tf.keras.layers.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(2, (3,3), activation=tf.nn.relu)(input_)
    #x = tf.keras.layers.Lambda(lambda x: tf.nn.avg_pool(x, (1,2,2,1), (1,2,2,1),
    #                                                    "VALID"))(x)
    x = tf.keras.layers.Conv2D(2, (5,5), strides=(2,2))(x)
    x = tf.keras.layers.Conv2D(2, (3,3), activation=tf.nn.relu)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(10, activation=tf.nn.relu)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    output_ = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(x)

    model = tf.keras.Model(inputs=input_, outputs=output_)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())
    hist = model.fit(x_train, y_train, epochs=2)
    acc[0] += hist.history['acc'][0]
    acc[1] += hist.history['acc'][1]
    tf.keras.backend.clear_session()
print('overall accuracy is {} for first epoch and {} for second epoch'.format(
    acc[0]/iterations, acc[1]/iterations), file=f, end='\n')