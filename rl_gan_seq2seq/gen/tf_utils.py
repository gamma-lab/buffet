from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import math


from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs



# pylint: disable=protected-access,invalid-name
RNNCell = rnn_cell_impl.RNNCell

def sum(x, axis=None, keepdims=False):
    axis = None if axis is None else [axis]
    return tf.reduce_sum(x, axis=axis, keep_dims=keepdims)


def mean(x, axis=None, keepdims=False):
    axis = None if axis is None else [axis]
    return tf.reduce_mean(x, axis=axis, keep_dims=keepdims)


def var(x, axis=None, keepdims=False):
    meanx = mean(x, axis=axis, keepdims=keepdims)
    return mean(tf.square(x - meanx), axis=axis, keepdims=keepdims)


def std(x, axis=None, keepdims=False):
    return tf.sqrt(var(x, axis=axis, keepdims=keepdims))


def max(x, axis=None, keepdims=False):
    axis = None if axis is None else [axis]
    return tf.reduce_max(x, axis=axis, keep_dims=keepdims)


def min(x, axis=None, keepdims=False):
    axis = None if axis is None else [axis]
    return tf.reduce_min(x, axis=axis, keep_dims=keepdims)


def concatenate(arrs, axis=0):
    return tf.concat(axis=axis, values=arrs)


def argmax(x, axis=None):
    return tf.argmax(x, axis=axis)


def policy_entropy(logits):
    a0 = logits - max(logits, axis=-1, keepdims=True)
    ea0 = tf.exp(a0)
    z0 = sum(ea0, axis=-1, keepdims=True)
    p0 = ea0 / z0
    return sum(p0 * (tf.log(z0) - a0), axis=-1)

class EmbeddingWrapper(RNNCell):
  def __init__(self,
               cell,
               embedding_classes,
               embedding_size,
               initializer=None,
               reuse=None):
    """Create a cell with an added input embedding.
    Args:
      cell: an RNNCell, an embedding will be put before its inputs.
      embedding_classes: integer, how many symbols will be embedded.
      embedding_size: integer, the size of the vectors we embed into.
      initializer: an initializer to use when creating the embedding;
        if None, the initializer from variable scope or a default one is used.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
    Raises:
      TypeError: if cell is not an RNNCell.
      ValueError: if embedding_classes is not positive.
    """
    super(EmbeddingWrapper, self).__init__(_reuse=reuse)
    self._cell = cell
    self._embedding_classes = embedding_classes
    self._embedding_size = embedding_size
    self._initializer = initializer

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def zero_state(self, batch_size, dtype):
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      return self._cell.zero_state(batch_size, dtype)

  def call(self, inputs, state):
    """Run the cell on embedded inputs."""
    with ops.device("/gpu:0"):
      if self._initializer:
        initializer = self._initializer
      elif vs.get_variable_scope().initializer:
        initializer = vs.get_variable_scope().initializer
      else:
        # Default initializer for embeddings should have variance=1.
        sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
        initializer = init_ops.random_uniform_initializer(-sqrt3, sqrt3)

      if isinstance(state, tuple):
        data_type = state[0].dtype
      else:
        data_type = state.dtype

      embedding = vs.get_variable(
          "embedding", [self._embedding_classes, self._embedding_size],
          initializer=initializer,
          dtype=data_type)
      embedded = embedding_ops.embedding_lookup(embedding,
                                                array_ops.reshape(inputs, [-1]))

      return self._cell(embedded, state)