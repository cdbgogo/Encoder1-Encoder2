# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Defines the Transformer model, and its encoder and decoder stacks.
Model paper: https://arxiv.org/pdf/1706.03762.pdf
Transformer model code source: https://github.com/tensorflow/tensor2tensor
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-bad-import-order

from .transformer_utils import attention_layer
from .transformer_utils import ffn_layer
from .transformer_utils import model_utils

_NEG_INF = -1e9


class Transformer(object):
    """Transformer model for sequence to sequence data.
    Implemented as described in: https://arxiv.org/pdf/1706.03762.pdf
    The Transformer model consists of an encoder and decoder. The input is an int
    sequence (or a batch of sequences). The encoder produces a continous
    representation, and the decoder uses the encoder output to generate
    probabilities for the output sequence.
    """

    def __init__(self, params, train):
        """Initialize layers to build Transformer model.
        Args:
          params: hyperparameter object defining layer sizes, dropout values, etc.
          train: boolean indicating whether the model is in training mode. Used to
            determine if dropout layers should be added.
        """
        self.train = train
        self.params = params
        self.encoder_stack = EncoderStack(params, train)

    def __call__(self, inputs, embedded_words):
        """Calculate target logits or inferred target sequences.
        Args:
          inputs: int tensor with shape [batch_size, input_length].
          embedded_words: [batch_size, input_length. emb_size]
          targets: None or int tensor with shape [batch_size, target_length].
        Returns:
            encoder_outputs: [batch_size, input_length, hidden_size]
        """
        # Variance scaling is used here because it seems to work in many problems.
        # Other reasonable initializers may also work just as well.
        initializer = tf.variance_scaling_initializer(
            self.params["initializer_gain"], mode="fan_avg", distribution="uniform")
        with tf.variable_scope("Transformer", initializer=initializer):
            # Calculate attention bias for encoder self-attention and decoder
            # multi-headed attention layers.
            attention_bias = model_utils.get_padding_bias(inputs)  # [batch_size, 1, 1, length]
            # print("attention_bias: {}".format(attention_bias.shape))

            # Run the inputs through the encoder layer to map the symbol
            # representations to continuous representations.
            encoder_outputs = self.encode(inputs, embedded_words, attention_bias)
            return encoder_outputs

    def encode(self, inputs, embedded_words, attention_bias):
        """Generate continuous representation for inputs.
        Args:
          inputs: int tensor with shape [batch_size, input_length].
          attention_bias: float tensor with shape [batch_size, 1, 1, input_length]
        Returns:
          float tensor with shape [batch_size, input_length, hidden_size]
        """
        with tf.name_scope("encode"):
            # Prepare inputs to the layer stack by adding positional encodings and
            # applying dropout.
            # embedded_inputs = self.embedding_softmax_layer(inputs)
            inputs_padding = model_utils.get_padding(inputs)

            with tf.name_scope("add_pos_encoding"):
                length = tf.shape(embedded_words)[1]
                pos_encoding = model_utils.get_position_encoding(
                    length, self.params["hidden_size"])
                encoder_inputs = embedded_words + pos_encoding

            # if self.train:
            #     encoder_inputs = tf.nn.dropout(
            #         encoder_inputs, 1 - self.params["layer_postprocess_dropout"])
            encoder_inputs = tf.layers.dropout(encoder_inputs, self.params["layer_postprocess_dropout"], training=self.train)

            return self.encoder_stack(encoder_inputs, attention_bias, inputs_padding)


class LayerNormalization(tf.layers.Layer):
    """Applies layer normalization."""

    def __init__(self, hidden_size):
        super(LayerNormalization, self).__init__()
        self.hidden_size = hidden_size

    def build(self, _):
        self.scale = tf.get_variable("layer_norm_scale", [self.hidden_size],
                                     initializer=tf.ones_initializer())
        self.bias = tf.get_variable("layer_norm_bias", [self.hidden_size],
                                    initializer=tf.zeros_initializer())
        self.built = True

    def call(self, x, epsilon=1e-6):
        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
        norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
        return norm_x * self.scale + self.bias


class PrePostProcessingWrapper(object):
    """Wrapper class that applies layer pre-processing and post-processing."""

    def __init__(self, layer, params, train):
        self.layer = layer
        self.postprocess_dropout = params["layer_postprocess_dropout"]
        self.train = train

        # Create normalization layer
        self.layer_norm = LayerNormalization(params["hidden_size"])

    def __call__(self, x, *args, **kwargs):
        # Preprocessing: apply layer normalization
        y = self.layer_norm(x)

        # Get layer output
        y = self.layer(y, *args, **kwargs)

        # Postprocessing: apply dropout and residual connection
        # if self.train:
        #     y = tf.nn.dropout(y, 1 - self.postprocess_dropout)

        y = tf.layers.dropout(y, self.postprocess_dropout, training=self.train)

        return x + y


class EncoderStack(tf.layers.Layer):
    """Transformer encoder stack.
    The encoder stack is made up of N identical layers. Each layer is composed
    of the sublayers:
      1. Self-attention layer
      2. Feedforward network (which is 2 fully-connected layers)
    """

    def __init__(self, params, train):
        super(EncoderStack, self).__init__()
        self.layers = []
        for _ in range(params["num_hidden_layers"]):
            # Create sublayers for each layer.
            self_attention_layer = attention_layer.SelfAttention(
                params["hidden_size"], params["num_heads"],
                params["attention_dropout"], train)
            feed_forward_network = ffn_layer.FeedFowardNetwork(
                params["hidden_size"], params["filter_size"],
                params["relu_dropout"], train, params["allow_ffn_pad"])

            self.layers.append([
                PrePostProcessingWrapper(self_attention_layer, params, train),
                PrePostProcessingWrapper(feed_forward_network, params, train)])

        # Create final layer normalization layer.
        self.output_normalization = LayerNormalization(params["hidden_size"])

    def call(self, encoder_inputs, attention_bias, inputs_padding):
        """Return the output of the encoder layer stacks.
        Args:
          encoder_inputs: tensor with shape [batch_size, input_length, hidden_size]
          attention_bias: bias for the encoder self-attention layer.
            [batch_size, 1, 1, input_length]
          inputs_padding: P
        Returns:
          Output of encoder layer stack.
          float32 tensor with shape [batch_size, input_length, hidden_size]
        """
        for n, layer in enumerate(self.layers):
            # Run inputs through the sublayers.
            self_attention_layer = layer[0]
            feed_forward_network = layer[1]

            with tf.variable_scope("layer_%d" % n):
                with tf.variable_scope("self_attention"):
                    encoder_inputs = self_attention_layer(encoder_inputs, attention_bias)
                with tf.variable_scope("ffn"):
                    encoder_inputs = feed_forward_network(encoder_inputs, inputs_padding)

        return self.output_normalization(encoder_inputs)


