# 2018, Patrick Wieschollek <mail@patwie.com>

# manually generated file
import tensorflow as tf
import os
from tensorflow.python.framework import ops

__all__ = []

def load_op(name, has_grad=False):
  """Load operation and add it to __all__ for imports.

  Args:
      name (str): name of operation without "_op" suffix
      has_grad (bool, optional): gradient (if exists) should be loaded as well

  Returns:
      functions
  """
  global __all__
  path = os.path.join(os.path.dirname(__file__), '%s_op.so' % name)
  _module = tf.load_op_library(path)
  if has_grad:
    __all__.append('%s' % name)
    __all__.append('%s_grad' % name)
    return getattr(_module, '%s' % name), getattr(_module, '%s_grad' % name)
  else:
    __all__.append('%s' % name)
    return getattr(_module, '%s' % name)


ft_inverse, ft_inverse_grad = load_op('ft_inverse', has_grad=True)


@ops.RegisterGradient("FtInverse")
def _FtInverseGrad(op, *grads):
  stride = op.get_attr('stride')
  pool_size = op.get_attr('pool_size')
  x = op.inputs[0]
  grad_in = grads[0]
  return ft_inverse_grad(x=x, gradients=grad_in, stride=stride, pool_size=pool_size)
