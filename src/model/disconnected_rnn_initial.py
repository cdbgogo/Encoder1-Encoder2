#!/usr/preprocess/env python
# -*- coding: utf-8 -*-
"""
Author	:

Date	:

Brief	:
"""

import tensorflow as tf
import numpy as np
from collections import defaultdict


try:
    from . import layer
    from . import transformer
    from . import utils_fast_disa
except:
    from src.model import layer
    from src.model import transformer
    from src.model import utils_fast_disa


class Disconnected_RNN:
    'Disconnected Recurrent Neural Networks for Text Categorization'
    def __init__(self, config, embedding=None):
        self._config = config
        self.emb_size = config.emb_size
        self.vocab_size = config.vocab_size
        self.n_classes = config.n_classes
        self.rnn_units = config.rnn_units
        self.window_size = config.window_size
        self.real_window_size = config.window_size
        self.mlp_units = config.mlp_units
        self.batch_size = config.batch_size

        self.filter_size = [3,5,7]
        self.filter_num = 128

        self.max_seq_len = config.max_sequence_length

        self.emb_initializer = tf.random_uniform_initializer(minval=-0.01, maxval=0.01)
        # self.emb_initializer = layer.xavier_initializer(self._config.xavier_factor, seed=config.fixedrng)
        #
        if embedding is None:
            self.embedding_table = tf.get_variable("embedding", shape=[self.vocab_size, self.emb_size],
                                                   initializer=self.emb_initializer)
        else:
            self.embedding_table = tf.get_variable(name="embedding", shape=[self.vocab_size, self.emb_size],
                                         initializer=tf.constant_initializer(embedding))
        print self.embedding_table.shape
        # Inputs
        self.sequence = tf.placeholder(tf.int32, [self.batch_size, self.max_seq_len], name='sequence')

        self.sequence_length = tf.placeholder(tf.int32, [self.batch_size, 1], name='sequence_length')
        self.label = tf.placeholder(tf.float32, [self.batch_size, None], name='label')

        self.is_train = tf.placeholder(tf.bool, name="is_train")  # for batch normalization

        self.rnn_keep_prob = tf.cond(self.is_train, lambda: tf.constant(1.0 - config.rnn_drop), lambda: tf.constant(1.0))
        self.fc_drop = config.fc_drop
        self.attention_drop = config.attention_drop

        if config.attention_type is not None:
            if config.encode_type is None:
                print("must choose an encode type")
                exit()
            elif config.encode_type == "transformer":
                params = self.get_param(config)
                self.encoder_func = transformer.Transformer(params=params, train=self.is_train)

        if self._config.feature_type == 'rnn':
            with tf.name_scope('feature_extractor_rnn'):
                # self.rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=self.rnn_units,
                #                                        kernel_initializer=layer.xavier_initializer(
                #                                            self._config.xavier_factor, seed=config.fixedrng))
                self.rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=self.rnn_units,
                                                       kernel_initializer=tf.initializers.orthogonal())
                # self.rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=self.rnn_units)
                self.rnn_cell = tf.nn.rnn_cell.DropoutWrapper(self.rnn_cell, input_keep_prob=self.rnn_keep_prob,
                                                              output_keep_prob=self.rnn_keep_prob, state_keep_prob=1.0)
                self.fixedrng = np.random.RandomState(config.fixedrng)
                # self.u = layer.norm_weight(self.mlp_units, self.n_classes, scale=config.xavier_factor,
                #                            rng=self.fixedrng)  # y_dim is the class number
                # self.W = layer.norm_weight(self.mlp_units, self.mlp_units, scale=config.xavier_factor, rng=self.fixedrng)
                # self.WC = layer.norm_weight(self.mlp_units, self.mlp_units, scale=config.xavier_factor, rng=self.fixedrng)
                self.u = layer.weight_variable([self.mlp_units, self.n_classes], level="u", factor=config.xavier_factor)
                self.W = layer.weight_variable([self.mlp_units, self.mlp_units], level="w", factor=config.xavier_factor)
                self.WC = layer.weight_variable([self.mlp_units, self.mlp_units], level="wc", factor=config.xavier_factor)
            self.logits_op = self.logits_rnn()
        elif self._config.feature_type == 'cnn':
            self.logits_op = self.logits_cnn_1d()
        elif self._config.feature_type == "dpcnn":
            self.logits_op = self.logits_cnn_dp()
        else:
            raise NotImplementedError
        self.loss_op = self.loss()
        self.result_op = self.result()
        self.loss_encode_op = self.loss_encoder()

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

    def pre_pad(self, x):
        '''
        pre pad for d_rnn
        :param x: A 3d tensor with shape of [batch_size, sequence_length, emb_size]
        :return: A 4d tensor for d_rnn with shape of [batch_size, block, window_size, emb_size]
        '''
        pad_input = tf.pad(x, [[0, 0], [self.window_size - 1, 0], [0, 0]], mode="CONSTANT")
        # print("pad_input:", pad_input.get_shape()) (batch_size, seq_max_len + window_size - 1, embed_size)
        # self.tmp1 = pad_input
        rnn_inputs = []
        for i in range(self.max_seq_len):
            rnn_inputs.append(tf.slice(pad_input, [0, i, 0], [-1, self.window_size, -1], name='rnn_input'))
        rnn_input_tensor = tf.stack(rnn_inputs, 1)  # (batch_size, seq_max_len, window_size, embed_size)

        # if self._config.attention_type in ["same_init", "diff_init"]:
        #     # self.global_fake: [self.batch_size, self.max_seq_len, self.emb_size])
        #     fake_input = tf.reshape(self.global_fake, [self.batch_size, self.max_seq_len, 1, self.emb_size])
        #     rnn_input_tensor = tf.concat([fake_input, rnn_input_tensor], 2)
        #     self.real_window_size = self.window_size + 1
        # elif self._config.attention_type in ["attend_init"]:
        #     block_rep = tf.reduce_mean(rnn_input_tensor, 2)  # (batch_size, seq_max_len, embed_size)
        #     fake_input = layer.basic_attention(block_rep, self.global_fake, "rnn")
        #     fake_input = tf.reshape(fake_input, [self.batch_size, self.max_seq_len, 1, self.emb_size])
        #     rnn_input_tensor = tf.concat([fake_input, rnn_input_tensor], 2)
        #     self.real_window_size = self.window_size + 1


        # # print("rnn_input_tensor:", rnn_input_tensor.get_shape())
        # self.tmp2 = rnn_input_tensor
        return rnn_input_tensor

    def maxpooling(self, x):
        # x: [batch_size, block, mlp_units]
        x = tf.reduce_max(x, axis=1)
        return tf.reshape(x, [-1, self.mlp_units])

    def conv1d(self, x, window_h, window_w, level="1", global_infor=None, no_act=False):
        # window_h: filter_size
        # window_w: represent_size
        # global_infor: [b_s, max_sl, emb_size]
        filter_shape = [window_h, window_w, self.filter_num]
        conv_W = layer.conv_weight_variable(filter_shape, name=level)
        conv_b = layer.bias_variable([self.filter_num])
        conv = tf.nn.conv1d(x,
                            conv_W,
                            stride=1,
                            padding='SAME',
                            name='conv')
        # conv: [b_s, max_sl, filter_num]
        if global_infor is not None:
            filter_shape = [1, self.emb_size, self.filter_num]
            conv_g_W = layer.conv_weight_variable(filter_shape, name=level)
            conv_g_b = layer.bias_variable([self.filter_num])
            conv_g = tf.nn.conv1d(global_infor,
                                conv_g_W,
                                stride=1,
                                padding='SAME',
                                name='conv_g')
            conv = conv + tf.nn.bias_add(conv_g, conv_g_b)
        if no_act:
            h = tf.nn.bias_add(conv, conv_b)
        else:
            h = tf.nn.relu(tf.nn.bias_add(conv, conv_b), name='relu')
        return h

    def cnn_enc(self, x):
        concat_vec = []
        # for i in self.filter_size:
        for i in [3]:
            conved = self.conv1d(x, i, self.emb_size, level=i)  # (b_s, max_seq_len, filter_num)
            # conved = tf.reduce_max(conved, axis=1)
            concat_vec.append(conved)
        return tf.concat(concat_vec, -1)

    def d_rnn(self, drnn_input):
        # embedded_words: [batch_size, block, window_size, emb_size]
        drnn_input_reshape = tf.reshape(drnn_input, [-1, self.real_window_size, self.represent_size])
        # drnn_input_reshape: [batch_size * block, window_size, emb_size]
        if self.initial_state is None:
            print("############zero state#########")
            self.initial_state = self.rnn_cell.zero_state(self.batch_size * self.max_seq_len, dtype=tf.float32)
        # outputs, _ = tf.nn.dynamic_rnn(cell=self.rnn_cell, initial_state=self.initial_state,
        #                                dtype=tf.float32, inputs=drnn_input_reshape)
        # last_output = outputs[:, -1, :]  # [batch_size * block, 1, rnn_units]

        x = tf.unstack(drnn_input_reshape, self.real_window_size, 1)
        outputs, last_state = tf.nn.static_rnn(cell=self.rnn_cell, dtype=tf.float32, inputs=x,
                                                     initial_state=self.initial_state)
        last_output = outputs[-1]

        drnn_output = tf.reshape(last_output, [self.batch_size, -1, self.rnn_units])  # [batch_size, block, rnn_units]
        drnn_output = tf.layers.batch_normalization(drnn_output, training=self.is_train)
        return drnn_output

    def encoder(self, sequence, embedded_words):
        # return: (b_s, max_seq_len, global_encode_units)
        last_state = None
        global_encode = None
        with tf.variable_scope('encode_module'):
            if self._config.encode_type == "transformer":
                global_encode = self.encoder_func(sequence, embedded_words)
                global_encode_units = self.emb_size
            elif self._config.encode_type == "other_transformer":
                global_encode = layer.attention_fun(
                    embedded_words, dropout_rate=self.attention_drop,
                    is_training=self.is_train, config=self._config, scope="attention_encode")
                global_encode_units = self.emb_size
            elif self._config.encode_type == "disan":
                sequence_length = tf.squeeze(self.sequence_length)
                rep_mask = tf.sequence_mask(sequence_length, self._config.max_sequence_length)
                global_encode = utils_fast_disa.fast_directional_self_attention(
                    embedded_words, rep_mask, hn=self._config.disan_units,
                    head_num=self._config.attention_head, msl=self.max_seq_len,
                    is_train=self.is_train)
                global_encode_units = self._config.disan_units
            elif self._config.encode_type == "cnn":
                global_encode = self.cnn_enc(embedded_words)
                global_encode_units = 128
                print "encode is cnn"
            elif self._config.encode_type == "rnn":
                rnn_encode_cell = tf.nn.rnn_cell.GRUCell(name="encode_gru", num_units=self.rnn_units,
                                                         kernel_initializer=tf.initializers.orthogonal())
                # self.tmp2 = tf.get_variable("rnn/encode_gru/gates/bias:0")

                x = tf.unstack(embedded_words, self.max_seq_len, 1)
                global_encode, last_state = tf.nn.static_rnn(cell=rnn_encode_cell, dtype=tf.float32, inputs=x)
                # global_encode, last_state = tf.nn.dynamic_rnn(cell=rnn_encode_cell, dtype=tf.float32, inputs=embedded_words)
                # global_encode = tf.reduce_max(outputs, 1)
                # last_state = tf.get_variable("encode_v", [self.batch_size, self.rnn_units])
                global_encode_units = self.rnn_units
                print "*****encode is rnn*******"
            elif self._config.encode_type == "w":
                global_encode = layer.fc_fun(embedded_words, self.emb_size, initial_type=self._config.initial_type,
                                             factor=self._config.xavier_factor, activation="leaky_relu")
                global_encode_units = self.emb_size
                print "encode is w"
            else:
                raise NotImplementedError
            self.global_encode = global_encode
        return global_encode, global_encode_units, last_state

    def logits_rnn(self):
        self.represent_size = self.emb_size
        self.initial_state = None
        # self.initial_state = self.rnn_cell.zero_state(self.batch_size*self.max_seq_len, dtype=tf.float32)
        # self.initial_state = tf.cast(self.initial_state, tf.float32)

        embedded_words = self.embedding(self.sequence)
        # embedded_words = tf.layers.dropout(embedded_words, self.fc_drop, training=self.is_train)
        if self._config.global_size:
            global_size = self._config.global_size
        else:
            global_size = self.rnn_units
        if self._config.attention_type == "pre_attention":
            global_encode, global_encode_units, last_state = self.encoder(self.sequence, embedded_words)
            embedded_words = global_encode
        elif self._config.attention_type == "diff_concat":
            global_encode, global_encode_units, last_state = self.encoder(self.sequence, embedded_words)
            global_encode_mlp = layer.fc_fun(
                global_encode, global_size, initial_type=self._config.initial_type)
            self.represent_size += global_size
            embedded_words = tf.concat([embedded_words, global_encode_mlp], axis=-1)
        elif self._config.attention_type == "same_concat":
            global_encode, global_encode_units, last_state = self.encoder(self.sequence, embedded_words)
            global_encode_mlp = layer.fc_fun(
                global_encode, global_size, initial_type=self._config.initial_type)
            global_encode_mlp = tf.reduce_max(global_encode_mlp, axis=1)
            global_encode_mlp = tf.reshape(tf.tile(global_encode_mlp, [1, self.max_seq_len]), [self.batch_size, self.max_seq_len, -1])
            self.represent_size += global_size
            print self.represent_size
            embedded_words = tf.concat([embedded_words, global_encode_mlp], axis=-1)
        elif self._config.attention_type == "same_init":
            global_encode, global_encode_units, last_state = self.encoder(self.sequence, embedded_words)
            if self._config.encode_type == "rnn":
                last_state = last_state
                # last_state = layer.fc_fun(
                #     last_state, self.rnn_units, initial_type=self._config.initial_type,
                #     factor=self._config.xavier_factor, activation="relu")
            elif self._config.encode_type == "cnn":
                last_state = tf.reduce_max(global_encode, 1)
                last_state = tf.layers.batch_normalization(last_state, training=self.is_train)
                last_state = layer.fc_fun(
                    last_state, self.rnn_units, initial_type=self._config.initial_type, activation="relu")
            print ("********initial state*********")
            last_state = tf.tile(last_state, [1, self.max_seq_len])
            # last_state = tf.reshape(last_state, [self.batch_size, self.rnn_units, self.max_seq_len])
            last_state = tf.reshape(last_state, [self.batch_size * self.max_seq_len, self.rnn_units])
            print ("initial_state.shape {}".format(last_state.shape))
            self.initial_state = last_state
            # self.initial_state = tf.get_variable(name='initial_state', shape=[self.batch_size*self.max_seq_len, self.rnn_units])
        elif self._config.attention_type in ["diff_init", "attend_init"]:
            global_encode, global_encode_units, last_state = self.encoder(self.sequence, embedded_words)
            global_encode_mlp = layer.fc_fun(
                global_encode, self.rnn_units, initial_type=self._config.initial_type)
            self.global_fake = tf.reshape(global_encode_mlp, [self.batch_size, self.max_seq_len, self.emb_size])
            #  even if we set self.initial_state as a variable, it stills dons't update, for there is no grad in the variable
            # init_state = tf.get_variable(name='initial_state', shape=[self.batch_size*self.max_seq_len, self.rnn_units])
        else:
            if self._config.attention_type is not None:
                raise NotImplementedError

        input_pad = self.pre_pad(embedded_words)  # [batch_size, block, window_size, emb_size]
        drnn_output = self.d_rnn(input_pad)  # [batch_size, block, rnn_units]

        # self.tmp2 = drnn_output
        drnn_output = tf.reshape(drnn_output, [-1, self.mlp_units])
        drnn_output = tf.matmul(drnn_output, self.WC)
        drnn_output = tf.reshape(drnn_output, [self.batch_size, -1, self.mlp_units])

        mask = tf.sequence_mask(self.sequence_length, self.max_seq_len, dtype=drnn_output.dtype)  # [batch_size, max_seq_len]
        mask = tf.reshape(mask, [self.batch_size, self.max_seq_len, 1])
        # self.tmp1 = mask
        drnn_output = drnn_output * mask

        hs = tf.reduce_max(drnn_output, axis=1)
        # hs = tf.layers.dropout(hs, self.fc_drop, training=self.is_train)

        mlp = tf.matmul(hs, self.W)

        mlp = tf.layers.batch_normalization(mlp, training=self.is_train)
        mlp = tf.nn.relu(mlp)

        # mlp = tf.layers.dropout(mlp, self.fc_drop, training=self.is_train)
        fcl_output = tf.matmul(mlp, self.u)
        # self.tmp2 = fcl_output

        return fcl_output

    def cnn(self, x, global_infor):
        concat_vec = []
        for filter_i in self.filter_size:
            if self._config.attention_type == "attend_init":
                pad_input = tf.pad(x, [[0, 0], [filter_i - 1, 0], [0, 0]], mode="CONSTANT")
                # print("pad_input:", pad_input.get_shape()) (batch_size, seq_max_len + window_size - 1, embed_size)
                # self.tmp1 = pad_input
                cnn_blocks = []
                for tmp_i in range(self.max_seq_len):
                    cnn_blocks.append(tf.slice(pad_input, [0, tmp_i, 0], [-1, filter_i, -1], name='cnn_block'))
                cnn_blocks = tf.stack(cnn_blocks, 1)  # (batch_size, seq_max_len, filter_i, embed_size)
                cnn_blocks = tf.reduce_mean(cnn_blocks, 2)
                global_infor = layer.basic_attention(cnn_blocks, global_infor, filter_i)
            conved = self.conv1d(x, filter_i, self.represent_size, global_infor)
            conved = tf.reduce_max(conved, axis=1)
            concat_vec.append(conved)
        return tf.concat(concat_vec, -1)

    def logits_cnn_1d(self):
        embedded_words = self.embedding(self.sequence)
        embedded_words = tf.layers.dropout(embedded_words, self.fc_drop, training=self.is_train)
        if self._config.global_size:
            global_size = self._config.global_size
        else:
            global_size = self.rnn_units
        self.global_fake = None
        self.represent_size = self.emb_size
        if self._config.attention_type == "pre_attention":
            global_encode, global_encode_units = self.encoder(self.sequence, embedded_words)
            embedded_words = global_encode
        elif self._config.attention_type == "diff_concat":
            global_encode, global_encode_units = self.encoder(self.sequence, embedded_words)
            global_encode_mlp = layer.fc_fun(
                global_encode, global_size, initial_type=self._config.initial_type,
                factor=self._config.xavier_factor)
            self.represent_size += global_size
            print("represent_size: {}".format(self.represent_size))
            embedded_words = tf.concat([embedded_words, global_encode_mlp], axis=-1)
        elif self._config.attention_type == "same_concat":
            global_encode, global_encode_units = self.encoder(self.sequence, embedded_words)
            global_encode_mlp = layer.fc_fun(
                global_encode, global_size, initial_type=self._config.initial_type,
                factor=self._config.xavier_factor)
            global_encode_mlp = tf.reduce_max(global_encode_mlp, axis=1)
            global_encode_mlp = tf.reshape(tf.tile(global_encode_mlp, [1, self.max_seq_len]), [self.batch_size, self.max_seq_len, -1])
            self.represent_size += global_size
            print("represent_size: {}".format(self.represent_size))
            embedded_words = tf.concat([embedded_words, global_encode_mlp], axis=-1)
        elif self._config.attention_type == "same_init":
            global_encode, global_encode_units = self.encoder(self.sequence, embedded_words)
            global_encode = tf.reduce_max(global_encode, axis=1)
            global_encode_mlp = layer.fc_fun(
                global_encode, self.rnn_units, initial_type=self._config.initial_type)
            global_encode_mlp = tf.tile(global_encode_mlp, [1, self.max_seq_len])
            global_encode_mlp = tf.reshape(global_encode_mlp, [self.batch_size, self.max_seq_len, self.emb_size])
            self.global_fake = global_encode_mlp
        elif self._config.attention_type in ["diff_init", "attend_init"]:
            global_encode, global_encode_units = self.encoder(self.sequence, embedded_words)
            global_encode_mlp = layer.fc_fun(
                global_encode, self.rnn_units, initial_type=self._config.initial_type)
            self.global_fake = tf.reshape(global_encode_mlp, [self.batch_size, self.max_seq_len, self.emb_size])
        else:
            if self._config.attention_type is not None:
                raise NotImplementedError
        outputs = self.cnn(embedded_words, global_infor=self.global_fake)

        outputs = tf.nn.leaky_relu(outputs)
        outputs = tf.layers.batch_normalization(outputs, training=self.is_train)

        fcl_output = layer.fc_fun(outputs, 2000, initial_type='xavier',
                                  activation=self._config.fc_activation_1)
        fcl_output = tf.layers.dropout(fcl_output, rate=self.fc_drop, training=self.is_train)
        fcl_output = layer.fc_fun(fcl_output, self.n_classes, initial_type='xavier')
        return fcl_output

    def logits_cnn_dp(self):
        print "DPCNN"
        with tf.name_scope("embedding"):
            self.filter_num = 250
            self.kernel_size = 3
            embedding = self.embedding(self.sequence)
            self.embedding_dim = self.emb_size
            # embedding_inputs = tf.expand_dims(embedding, axis=-1)  # [None,seq,embedding,1]
            # region_embedding  # [batch,seq-3+1,1,250]
            # region_embedding = tf.layers.conv2d(embedding_inputs, self.num_filters,
            #                                     [self.kernel_size, self.embedding_dim])
            region_embedding = self.conv1d(embedding, 3, self.emb_size, no_act=True)
            region_embedding = tf.expand_dims(region_embedding, axis=2)
            # (4, 254, 1, 250), max_sl: 256

            pre_activation = tf.nn.relu(region_embedding, name='preactivation')
        with tf.name_scope("conv3_0"):
            conv3 = tf.layers.conv2d(pre_activation, self.filter_num, self.kernel_size,
                                     padding="same")
            conv3 = tf.layers.batch_normalization(conv3, training=self.is_train)

        with tf.name_scope("conv3_1"):
            conv3 = tf.layers.conv2d(conv3, self.filter_num, self.kernel_size,
                                     padding="same")
            conv3 = tf.layers.batch_normalization(conv3, training=self.is_train)

        # print conv3.shape
        # (4, 254, 1, 250)

        # resdul
        conv3 = conv3 + region_embedding

        for block in range(6):
            with tf.name_scope("block_{}".format(block)):
                with tf.name_scope("pool_1"):
                    pool = tf.pad(conv3, paddings=[[0, 0], [0, 1], [0, 0], [0, 0]])
                    # print "pool", pool.shape  # (4, 255, 1, 250)
                    pool = tf.nn.max_pool(pool, [1, 3, 1, 1], strides=[1, 2, 1, 1], padding='VALID')
                    # print "pool", pool.shape  # (4, 127, 1, 250)

                with tf.name_scope("conv3_2"):
                    conv3 = tf.layers.conv2d(pool, self.filter_num, self.kernel_size,
                                             padding="same", activation=tf.nn.relu)
                    # print conv3.shape   # (4, 127, 1, 250)
                    conv3 = tf.layers.batch_normalization(conv3, training=self.is_train)

                with tf.name_scope("conv3_3"):
                    conv3 = tf.layers.conv2d(conv3, self.filter_num, self.kernel_size,
                                             padding="same", activation=tf.nn.relu)
                    # print conv3.shape  # (4, 127, 1, 250)
                    conv3 = tf.layers.batch_normalization(conv3, training=self.is_train)
                # resdul
                conv3 = conv3 + pool
            # print conv3.shape
        # pool_size = int((self.max_seq_len - 3 + 1) / 2)
        # conv3 = tf.layers.max_pooling1d(tf.squeeze(conv3, [2]), pool_size, 1)
        conv3 = tf.reduce_max(conv3, 1)

        conv3 = tf.squeeze(conv3)  # [batch,250]
        conv3 = tf.layers.dropout(conv3, self.fc_drop, training=self.is_train)
        fcl_output = layer.fc_fun(conv3, self.n_classes, initial_type='xavier')
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
        if self._config.l2_loss:
            print "use l2_loss"
            tv = tf.trainable_variables()
            regularization_cost = tf.add_n([tf.nn.l2_loss(v) for v in tv if 'bias' not in v.name]) * 0.001
            loss += regularization_cost
        return loss

    def loss_encoder(self):
        if self._config.attention_type is None\
                or (not self._config.encoder_fixed_epoch):
            return None
        label = self.label
        global_logits = tf.reduce_max(self.global_encode, axis=1)
        logits = layer.fc_fun(global_logits, self.n_classes)
        if self._config.type == 'single_label':
            global_loss = tf.reduce_sum(
                tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label)
            )
        elif self._config.type == 'multi_label':
            global_loss = tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=label)
            )
        else:
            raise NotImplementedError
        return global_loss

    def result(self):
        correct_prediction = tf.count_nonzero(tf.equal(tf.argmax(self.logits_op, 1), tf.argmax(self.label, 1)))
        return correct_prediction


def preprocess_for_drnn(sentences, config):
    '''
    preprocess for the disconnected rnn
    :param sentence: [sen1, sen2]
    :param config:
    :return: [[block1, block2...], [block1, block2]]
    '''
    # print(sentences)
    window_size = config.model.window_size
    sentence_block_all = []
    for sentence in sentences:
        # print(sentence)
        sentence_block = []
        index = 0
        while True:
            sentence_block.append([sentence[index: index + window_size]])
            if index + window_size >= len(sentence):
                break
            index += 1
        sentence_block_all.append(sentence_block)
    sentence_block_all = np.asarray(sentence_block_all).squeeze(2)
    return sentence_block_all




