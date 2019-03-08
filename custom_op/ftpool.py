from __future__ import print_function
import numpy as np
import tensorflow as tf
from user_ops import matrix_add

np.random.seed(42)
tf.set_random_seed(42)

matA = np.array(
[
[
[[10],[10],[8],[8]],
[[10],[10],[8],[8]],
[[2],[2],[6],[6]],
[[2],[2],[6],[6]],
[[4],[4],[1],[1]],
[[4],[4],[1],[1]]
]
]
, dtype=np.float32)
print(matA.shape)
print(matA[0, ..., 0])


A = tf.placeholder(tf.float32, shape=matA.shape)

actual_op = matrix_add(A,(2,2))


with tf.Session() as sess:
  a = (sess.run(actual_op, {A: matA}))

print(a.shape)
print(a[0, ..., 0])
