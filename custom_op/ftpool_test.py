import tensorflow as tf
import numpy as np
import os
from user_ops import ft_pool

tf.enable_eager_execution()
print('tf is executing eagerly: ', tf.executing_eagerly())

input = np.ones([1,5,5,1], dtype=np.float)
for i in range(5):
    input[0,i,i,0] = 2
output = ft_pool(input, (2,2), 10)

print("input shape", input.shape, "output shape", output.shape)
print("input tensor is \n", input[0, ..., 0])
print("output tensor is \n", output[0, ..., 0])