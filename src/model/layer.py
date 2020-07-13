from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
import logging
import math

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops


def weight_variable(shape, factor, level=1):
    # initial = np.random.uniform(-0.01, 0.01, shape)
    # initial = xavier_initializer()
    # conv_W = tf.Variable(initial, name='conv_{}_W'.format(level), dtype=tf.float32)
    # conv_W = tf.get_variable(name='conv_{}_W'.format(level), shape=shape,
    #                          initializer=xavier_initializer(factor=factor))
    conv_W = tf.get_variable(name='W_{}'.format(level), shape=shape,
                             initializer=variance_scaling_initializer())
    return conv_W


def conv_weight_variable(shape, name=1):
    initial = np.random.uniform(-0.01, 0.01, shape)
    conv_W = tf.Variable(initial, name='conv_{}_W'.format(name), dtype=tf.float32)
    return conv_W


# initialize bias in CNN.
def bias_variable(shape, name=1):
    initial = tf.zeros(shape=shape)
    conv_b = tf.Variable(initial, name='b_{}'.format(name), dtype=tf.float32)
    return conv_b


def get_fc_layer(units, initial_type='normal', activation=None, factor=6, name='1'):
    logging.info('fc {} activation is {}'.format(name, activation))
    activation_fun = get_activation(activation)

    if initial_type == 'normal':
        kernel_initializer = tf.random_normal_initializer(stddev=0.01)
    elif initial_type == 'xavier':
        # kernel_initializer = tf.contrib.layers.xavier_initializer()
        kernel_initializer = xavier_initializer(factor=factor)
    elif initial_type == 'random':
        kernel_initializer = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
    else:
        raise NotImplementedError
    logging.info("fc {} kernel initial is {}".format(name, initial_type))
    return tf.layers.Dense(units=units, kernel_initializer=kernel_initializer, activation=activation_fun)


def parametric_relu(_x):
    alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                       initializer=tf.constant_initializer(0.0),
                        dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5
    return pos + neg


def get_activation(activation=None):
    if activation == 'relu':
        activation_fun = tf.nn.relu
    elif activation == 'leaky_relu':
        activation_fun = tf.nn.leaky_relu
    elif activation == 'tanh':
        activation_fun = tf.nn.tanh
    elif activation == 'sigmoid':
        activation_fun = tf.nn.sigmoid
    elif activation == 'p_relu':
        activation_fun = parametric_relu
    elif activation == 'elu':
        activation_fun = tf.nn.elu
    elif activation is None or activation == 'None':
        activation_fun = None
    else:
        raise NotImplementedError
    return activation_fun


def fc_fun(inputs, units, initial_type='normal', activation=None, name='1'):
    logging.info('fc {} activation is {}'.format(name, activation))
    activation_fun = get_activation(activation)
    if initial_type == 'normal':
        kernel_initializer = tf.random_normal_initializer(stddev=0.01)
    elif initial_type == 'xavier':
        # kernel_initializer = tf.contrib.layers.xavier_initializer()
        # kernel_initializer = xavier_initializer(factor=factor)
        kernel_initializer = variance_scaling_initializer()
    elif initial_type == 'uniform':
        kernel_initializer = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
    else:
        raise NotImplementedError
    logging.info("fc {} kernel initial is {}".format(name, initial_type))
    return tf.layers.dense(inputs, units, kernel_initializer=kernel_initializer, activation=activation_fun)


def get_optimizer(optimizer_type, trainer_config):
    if optimizer_type == 'Adadelta':
        learning_rate, rho, epsilon = trainer_config.learning_rate, trainer_config.rho, trainer_config.epsilon
        optimizer = tf.train.AdadeltaOptimizer(learning_rate, rho, epsilon)
        logging.info("optimizer is Adadelta, learning_rate is {}, rho is {}, epsilon is {}".format(learning_rate, rho, epsilon))
    elif optimizer_type == "Nadam":
        learning_rate = trainer_config.learning_rate
        logging.info('optimizer is Nadam, learning_rate is {}'.format(learning_rate))
        optimizer = tf.contrib.opt.NadamOptimizer(learning_rate)
    else:
        learning_rate = trainer_config.learning_rate
        optimizer = tf.train.AdamOptimizer(learning_rate)
        if optimizer_type == 'Adam':
            logging.info('optimizer is Adam, learning_rate is {}'.format(learning_rate))
        else:
            logging.info('not indicate optimizer, defaupt is Adam, learning_rate is {}'.format(learning_rate))
    return optimizer


def xavier_initializer(factor=6.0, uniform=True, seed=1996, dtype=dtypes.float32):
    """Returns an initializer performing "Xavier" initialization for weights.
    This function implements the weight initialization from:
    Xavier Glorot and Yoshua Bengio (2010):
           [Understanding the difficulty of training deep feedforward neural
           networks. International conference on artificial intelligence and
           statistics.](
           http://www.jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)
    This initializer is designed to keep the scale of the gradients roughly the
    same in all layers. In uniform distribution this ends up being the range:
    `x = sqrt(6. / (in + out)); [-x, x]` and for normal distribution a standard
    deviation of `sqrt(2. / (in + out))` is used.
    Args:
    uniform: Whether to use uniform or normal distributed random initialization.
    seed: A Python integer. Used to create random seeds. See
          `tf.set_random_seed` for behavior.
    dtype: The data type. Only floating point types are supported.
    Returns:
    An initializer for a weight matrix.
    """
    factor = factor / 6
    # return initializers.variance_scaling_initializer(factor=factor, mode='FAN_AVG',
    #                                   uniform=uniform, seed=seed, dtype=dtype)
    return variance_scaling_initializer(factor=factor, mode='FAN_AVG', other=False, scale=factor,
                                      uniform=uniform, seed=seed, dtype=dtype)


def basic_attention(queries, keys, name):
    num_units = queries.shape[2].value
    with tf.variable_scope("attend_init_{}".format(name)):
        Q = tf.layers.dense(queries, num_units, use_bias=False)  #
        Q = tf.nn.tanh(Q)
        K = tf.layers.dense(keys, num_units, use_bias=False)  # [N, msl_k, C]
        K = tf.nn.tanh(K)
        # V = tf.layers.dense(keys, num_units)  # [N, msl_k, C]
        V = keys

        # Multiplication
        scores = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (N, msl_q, msl_k)

        # scores = scores / (K.get_shape().as_list()[-1] ** 0.5)

        # Activation
        scores = tf.nn.softmax(scores)  # (N, msl_q, msl_k)

        # Weighted sum
        outputs = tf.matmul(scores, V)  # ( N, msl_q, C)
        return outputs


def attention_fun(embedded_inputs, dropout_rate=0, config=None, scope="attention", reuse=False, is_training=True):
    num_units = embedded_inputs.get_shape().as_list()[-1]
    # keys += positional_encoding(keys, num_units=num_units, zero_pad=False, scale=False, scope="enc_pe")
    pos_encoding = get_position_encoding(config.max_sequence_length, num_units)
    encoder_inputs = embedded_inputs + pos_encoding

    num_heads = config.attention_head
    encoder_inputs = tf.layers.dropout(encoder_inputs, dropout_rate, training=is_training)
    activation_fun = get_activation(config.attention_activation)

    for i in range(config.attention_block):
        with tf.variable_scope("num_blocks_{}".format(i)):
            #  Multihead Attention
            encoder_inputs = multihead_attention(queries=encoder_inputs, keys=encoder_inputs, num_units=num_units, num_heads=num_heads, attention_activation=activation_fun,
                                       dropout_rate=dropout_rate, is_training=is_training, scope=scope, reuse=reuse)
            encoder_inputs = feedforward(encoder_inputs, num_units=[4*num_units, num_units])
    return encoder_inputs


def multihead_attention(queries, keys, num_units, num_heads=2, dropout_rate=0, residual=True,
                        attention_activation=None, scope="attention", reuse=False, is_training=True):
    '''Applies multi-head attention.

        Args:
          queries: A 3d tensor with shape of [N, T_q, C_q].
          keys: A 3d tensor with shape of [N, T_k, C_k].
          num_units: hidden units
          attention_activation: attention_fun
          dropout_rate: A floating point number.
          is_training: Boolean. Controller of mechanism for dropout.
          num_heads: An int. Number of heads.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.

        Returns
          A 3d tensor with shape of (N, T_q, c)
        '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=attention_activation)  # [N, sequence_length, C]
        K = tf.layers.dense(keys, num_units, activation=attention_activation)  # [N, sequence_length, C]
        V = tf.layers.dense(keys, num_units, activation=attention_activation)  # [N, sequence_length, C]

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Residual connection
        if residual:
            outputs += queries

        # Normalize
        outputs = normalize(outputs)  # (N, T_q, C)

        return outputs


def get_position_encoding(length, hidden_size, min_timescale=1.0, max_timescale=1.0e4):
    """Return positional encoding.
    by https://github.com/tensorflow/models/blob/master/official/transformer/model/model_utils.py

    Calculates the position encoding as a mix of sine and cosine functions with
    geometrically increasing wavelengths.
    Defined and formulized in Attention is All You Need, section 3.5.
    Args:
    length: Sequence length.
    hidden_size: Size of the
    min_timescale: Minimum scale that will be applied at each position
    max_timescale: Maximum scale that will be applied at each position
    Returns:
    Tensor with shape [length, hidden_size]
    """
    position = tf.to_float(tf.range(length))
    num_timescales = hidden_size // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    return signal


def positional_encoding(inputs,
                        num_units,
                        zero_pad=True,
                        scale=True,
                        scope="positional_encoding",
                        reuse=None):
    '''Sinusoidal Positional_Encoding.

    Args:
      # inputs: A 2d Tensor with shape of (N, T).
      inputs: A 3d Tensor with shape of (N, T, _).
      num_units: Output dimensionality
      zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
      scale: Boolean. If True, the output will be multiplied by sqrt num_units(check details from paper)
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
        A 'Tensor' with one more rank than inputs's, with the dimensionality should be 'num_units'
    '''
    N, T, h = inputs.get_shape().as_list()
    N = tf.shape(inputs)[0]
    # print(inputs.get_shape().as_list())
    with tf.variable_scope(scope, reuse=reuse):
        # position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), tf.stack([N, 1]))  # fix by hr
        # https: // stackoverflow.com / questions / 38806136 / tensorflow - shape - of - a - tiled - tensor

        # First part of the PE function: sin and cos argument
        # position_enc = np.array([
        #     [pos / np.power(10000, 2.*i/num_units) for i in range(num_units)]
        #     for pos in range(T)])
        position_enc = np.array([
            [pos / np.power(10000, (i - i % 2) / num_units) for i in range(num_units)]
            for pos in range(T)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        # Convert to a tensor
        lookup_table = tf.convert_to_tensor(position_enc)

        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

        if scale:
            outputs = outputs * num_units**0.5
        outputs = tf.cast(outputs, tf.float32)
        return outputs


def feedforward(inputs,
                num_units=[2048, 512],
                scope="multihead_attention",
                reuse=None):
    '''Point-wise feed forward net.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Residual connection
        outputs += inputs

        # Normalize
        outputs = normalize(outputs)

    return outputs


def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs


def ortho_weight(ndim, rng=np.random):
    W = rng.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u


def norm_weight(nin, nout=None, scale=0.1, ortho=False, rng=np.random):
    if ortho:
        return ortho_weight(nin, rng=rng)
    if nout:
        W = scale / np.sqrt(nin + nout) * rng.randn(nin, nout)
    else:
        W = scale / np.sqrt(nin) * rng.randn(nin)
    return tf.Variable(W, name='fcl_W', dtype=tf.float32)


def variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False, other=False, scale=1.0,
                                 seed=None, dtype=dtypes.float32):
    if not dtype.is_floating:
        raise TypeError('Cannot create initializer for non-floating point type.')
    if mode not in ['FAN_IN', 'FAN_OUT', 'FAN_AVG']:
        raise TypeError('Unknown mode %s [FAN_IN, FAN_OUT, FAN_AVG]', mode)

    # pylint: disable=unused-argument
    def _initializer(shape, dtype=dtype, partition_info=None):
        """Initializer function."""
        if not dtype.is_floating:
            raise TypeError('Cannot create initializer for non-floating point type.')
            # Estimating fan_in and fan_out is not possible to do perfectly, but we try.
            # This is the right thing for matrix multiply and convolutions.
        if shape:
            fan_in = float(shape[-2]) if len(shape) > 1 else float(shape[-1])
            fan_out = float(shape[-1])
        else:
            fan_in = 1.0
            fan_out = 1.0
        for dim in shape[:-2]:
            fan_in *= float(dim)
            fan_out *= float(dim)
        if mode == 'FAN_IN':
            # Count only number of input connections.
            n = fan_in
        elif mode == 'FAN_OUT':
            # Count only number of output connections.
            n = fan_out
        elif mode == 'FAN_AVG':
            # Average number of inputs and output connections.
            n = (fan_in + fan_out) / 2.0
        # print(fan_in, fan_out)
        if other:
            w = scale / np.sqrt(fan_in + fan_out)
            logging.info('in xavier, use other type, scale is {}'.format(scale))
            return w * random_ops.truncated_normal(shape, seed=seed)

        if uniform:
            # To get stddev = math.sqrt(factor / n) need to adjust for uniform.
            limit = math.sqrt(3.0 * factor / n)
            return random_ops.random_uniform(shape, -limit, limit,
                                       dtype, seed=seed)
        else:
            # To get stddev = math.sqrt(factor / n) need to adjust for truncated.
            trunc_stddev = math.sqrt(1.3 * factor / n)
            return random_ops.truncated_normal(shape, 0.0, trunc_stddev, dtype,
                                             seed=seed)
            # pylint: enable=unused-argument
    return _initializer