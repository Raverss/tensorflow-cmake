from __future__ import print_function
import numpy as np
import tensorflow as tf
from user_ops import matrix_add

np.random.seed(42)
tf.set_random_seed(42)

matA = np.array(
[
[
[[10],[10],[10],[10]],
[[10],[10],[10],[10]],
[[10],[8],[8],[10]],
[[10],[10],[10],[10]]
]
]
, dtype=np.float32)
print(matA.shape)


A = tf.placeholder(tf.float32, shape=[None, 4, 4, 1])

actual_op = matrix_add(A,(1.3,1.3))


with tf.Session() as sess:
  a = (sess.run(actual_op, {A: matA}))

print(a.shape)
print(a[0, ..., 0])
