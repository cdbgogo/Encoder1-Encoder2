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
try:
    from . import layer
    from . import utils_disan, utils_fast_disa
except:
    from src.model import layer
    from src.model import utils_disan, utils_fast_disa


class DiSAN:
    def __init__(self, config, embedding=None):
        self._config = config
        self.emb_size = config.emb_size
        self.vocab_size = config.vocab_size
        self.n_classes = config.n_classes
        self.mlp_units = config.mlp_units
        self.max_seq_len = config.max_sequence_length
        self.disan_units = config.disan_units


        # self.emb_initializer = tf.random_normal_initializer(stddev=0.1)
        if embedding is None:
            self.embedding_table = tf.get_variable("embedding", shape=[self.vocab_size, self.emb_size],)
                                                   # initializer=self.emb_initializer)
        else:
            self.embedding_table = tf.get_variable(name="embedding", shape=[self.vocab_size, self.emb_size],
                                         initializer=tf.constant_initializer(embedding))

        # Inputs
        self.sequence = tf.placeholder(tf.int32, [None, self.max_seq_len], name='sequence')
        self.sequence_length = tf.placeholder(tf.int32, [None, 1], name='sequence_length')
        self.batch_size = tf.shape(self.sequence)[0]

        self.label = tf.placeholder(tf.float32, [None, None], name='label')
        self.is_train = tf.placeholder(tf.bool, name='is_train')

        # self.fc_keep_prob = 1 - config.fc_drop
        self.fc_drop = config.fc_drop
        self.disan_keep = 1 - config.disan_drop

        # Fetch OPs
        self.logits_op = self.logits()
        self.loss_op = self.loss()
        self.result_op = self.result()


    def embedding(self, x):
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            embedded_words = tf.nn.embedding_lookup(self.embedding_table, x)
        return embedded_words

    def logits(self):
        embedded_words = self.embedding(self.sequence)
        sequence_length = tf.squeeze(self.sequence_length)
        rep_mask = tf.sequence_mask(sequence_length, self._config.max_sequence_length)
        if self._config.disan_type == 'origin':
            outputs = utils_disan.disan(embedded_words, rep_mask, self._config, is_train=self.is_train, keep_prob=self.disan_keep)
        elif self._config.disan_type == 'fast':
            # outputs, tmp1, tmp2 = utils_fast_disa.fast_directional_self_attention(embedded_words, rep_mask, hn=512, msl=self.max_seq_len, is_train=self.is_train)
            outputs = utils_fast_disa.fast_directional_self_attention(
                embedded_words, rep_mask, hn=self._config.disan_units,
                head_num=self._config.attention_head, msl=self.max_seq_len,
                is_train=self.is_train)
            outputs = tf.reduce_max(outputs, 1)
        else:
            raise NotImplementedError

        # self.tmp1= tmp1
        # self.tmp2 = tmp2
        # self.tmp1 = outputs

        # self.tmp2 = outputs
        # outputs = layer.fc_fun(tf.layers.flatten(embedded_words), self.n_classes)
        # return outputs
        fcl_output = layer.fc_fun(outputs, self.mlp_units, initial_type='xavier',
                                  activation=self._config.fc_activation_1)
        fcl_output = tf.layers.dropout(fcl_output, rate=self.fc_drop, training=self.is_train)
        fcl_output = layer.fc_fun(fcl_output, self.n_classes, initial_type='xavier')
        # self.tmp2 = fcl_output
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

