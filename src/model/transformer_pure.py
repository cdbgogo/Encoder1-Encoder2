#!/usr/preprocess/env python
# -*- coding: utf-8 -*-
"""
Author	:

Date	:

Brief	:
"""

import sys
import os
import tensorflow as tf
import numpy as np
from collections import defaultdict
try:
    from . import layer
    from . import transformer
except:
    from src.model import layer
    from src.model import transformer


class Transformer_pure:
    def __init__(self, config, embedding=None):
        self._config = config
        self.emb_size = config.emb_size
        self.vocab_size = config.vocab_size
        self.n_classes = config.n_classes
        self.max_seq_len = config.max_sequence_length
        self.mlp_units = config.mlp_units

        if embedding is None:
            self.embedding_table = tf.get_variable("embedding", shape=[self.vocab_size, self.emb_size])
        else:
            self.embedding_table = tf.get_variable(name="embedding", shape=[self.vocab_size, self.emb_size],
                                         initializer=tf.constant_initializer(embedding))

        # self.emb_initializer = tf.random_normal_initializer(stddev=0.1)
        # self.fc1 = layer.get_fc_layer(self.n_classes, initial_type='normal', activation=config.fc_activation_1)

        # Inputs
        self.sequence = tf.placeholder(tf.int32, [None, self.max_seq_len], name='sequence')
        self.sequence_length = tf.placeholder(tf.int32, [None, 1], name='sequence_length')

        self.label = tf.placeholder(tf.float32, [None, None], name='label')

        self.is_train = tf.placeholder(tf.bool, name='is_train')
        self.fc_drop = config.fc_drop

        params = self.get_param(config)
        self.transformer = transformer.Transformer(params=params, train=self.is_train)

        # Fetch OPs
        self.logits_op = self.logits()
        self.loss_op = self.loss()
        self.result_op = self.result()

    def get_param(self, config):
        BASE_PARAMS = defaultdict(
            lambda: None,  # Set default value to None.

            # Input params
            default_batch_size=2048,  # Maximum number of tokens per batch of examples.
            default_batch_size_tpu=32768,
            max_length=256,  # Maximum number of tokens per example.
            # max_length=256,  # Maximum number of tokens per example.

            # Model params
            initializer_gain=1.0,  # Used in trainable variable initialization.
            # vocab_size=33708,  # Number of tokens defined in the vocabulary file.
            hidden_size=512,  # Model dimension in the hidden layers.
            num_hidden_layers=6,  # Number of layers in the encoder and decoder stacks.
            # num_heads=8,  # Number of heads to use in multi-headed attention.
            num_heads=2,  # Number of heads to use in multi-headed attention.
            filter_size=2048,  # Inner layer dimension in the feedforward network.

            # Dropout values (only used when training)
            layer_postprocess_dropout=0.1,
            attention_dropout=0.1,
            relu_dropout=0.1,

            # Training params
            label_smoothing=0.1,
            learning_rate=2.0,
            learning_rate_decay_rate=1.0,
            learning_rate_warmup_steps=16000,

            # Optimizer params
            optimizer_adam_beta1=0.9,
            optimizer_adam_beta2=0.997,
            optimizer_adam_epsilon=1e-09,

            # TPU specific parameters
            use_tpu=False,
            static_batch=False,
            allow_ffn_pad=True,
        )
        BASE_PARAMS['max_length'] = config.max_sequence_length
        BASE_PARAMS['hidden_size'] = config.emb_size
        BASE_PARAMS['num_hidden_layers'] = config.attention_block
        BASE_PARAMS['num_heads'] = config.attention_head

        return BASE_PARAMS

    def embedding(self, x):
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            embedded_words = tf.nn.embedding_lookup(self.embedding_table, x)
        return embedded_words

    def logits(self):
        # x is fcl_output
        embedded_words = self.embedding(self.sequence)
        # outputs = self.transformer(self.sequence, embedded_words)
        outputs = self.transformer(self.sequence, embedded_words)
        # self.train_symbol_show = train_show

        outputs = tf.reduce_max(outputs, axis=1)

        fcl_output = layer.fc_fun(outputs, self.n_classes, initial_type='xavier')
        return fcl_output

    def loss(self):
        logits = self.logits_op
        label = self.label
        if self._config.type == 'single_label':
            loss = tf.reduce_sum(
                tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label)
            )
        elif self._config.type == 'multi_label':
            loss = tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=label)
            )
        else:
            raise NotImplementedError
        return loss

    def result(self):
        correct_prediction = tf.count_nonzero(tf.equal(tf.argmax(self.logits_op, 1), tf.argmax(self.label, 1)))
        return correct_prediction


