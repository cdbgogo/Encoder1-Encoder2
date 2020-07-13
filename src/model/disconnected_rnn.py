#!/usr/preprocess/env python
# -*- coding: utf-8 -*-
'''
The paper model is in here
'''

import tensorflow as tf
import numpy as np
from collections import defaultdict
import logging


try:
    from . import layer
    from . import transformer
    from . import utils_fast_disa
except:
    from src.model import layer
    from src.model import transformer
    from src.model import utils_fast_disa


class Disconnected_RNN:
    '''
    Encoder1-Enocoder2-Mode architecture
    Due to historical reasons, this class is named as Disconnected_RNN though it ensemble cnn/drnn/dpcnn as the encoder2
    '''
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
        if config.window_size is not None:
            self.filter_size = [config.window_size]
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
                logging.info("must choose an encode type")
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
                self.rnn_cell = tf.nn.rnn_cell.DropoutWrapper(self.rnn_cell, input_keep_prob=self.rnn_keep_prob,
                                                              output_keep_prob=self.rnn_keep_prob, state_keep_prob=1.0)
                self.fixedrng = np.random.RandomState(config.fixedrng)
                self.u = layer.weight_variable([self.mlp_units, self.n_classes], level="u", factor=config.xavier_factor)
                self.W = layer.weight_variable([self.mlp_units, self.mlp_units], level="w", factor=config.xavier_factor)
                self.WC = layer.weight_variable([self.mlp_units, self.mlp_units], level="wc", factor=config.xavier_factor)
            self.logits_op = self.logits_drnn()
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
        # get BERT param when we choose BERT as the encoder
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

        if self._config.attention_type in ["same_init", "diff_init"]:
            # self.global_fake: [self.batch_size, self.max_seq_len, self.emb_size])
            fake_input = tf.reshape(self.global_fake, [self.batch_size, self.max_seq_len, 1, self.emb_size])
            rnn_input_tensor = tf.concat([fake_input, rnn_input_tensor], 2)
            self.real_window_size = self.window_size + 1
        elif self._config.attention_type in ["attend_init"]:
            block_rep = tf.reduce_mean(rnn_input_tensor, 2)  # (batch_size, seq_max_len, embed_size)
            fake_input = layer.basic_attention(block_rep, self.global_fake, "rnn")
            fake_input = tf.layers.batch_normalization(fake_input, training=self.is_train)
            fake_input = layer.fc_fun(
                fake_input, self.emb_size, initial_type=self._config.initial_type, activation="relu")
            fake_input = tf.reshape(fake_input, [self.batch_size, self.max_seq_len, 1, self.emb_size])
            rnn_input_tensor = tf.concat([fake_input, rnn_input_tensor], 2)
            self.real_window_size = self.window_size + 1
        # print("rnn_input_tensor:", rnn_input_tensor.get_shape())
        # self.tmp2 = rnn_input_tensor
        return rnn_input_tensor

    def maxpooling(self, x):
        # x: [batch_size, block, mlp_units]
        x = tf.reduce_max(x, axis=1)
        return tf.reshape(x, [-1, self.mlp_units])

    def conv1d(self, x, window_h, window_w, level="1", global_infor=None, no_act=False, conv_W=None, conv_b=None):
        '''
        :param x:
        :param window_h: filter_size
        :param window_w: represent_size
        :param level:
        :param global_infor: [b_s, max_sl, emb_size]
        :param no_act:
        :param conv_W:
        :param conv_b:
        :return:
        '''
        filter_shape = [window_h, window_w, self.filter_num]
        if conv_W is None:
            conv_W = layer.conv_weight_variable(filter_shape, name=level)
        if conv_b is None:
            conv_b = layer.bias_variable([self.filter_num], name=level)
        conv = tf.nn.conv1d(x,
                            conv_W,
                            stride=1,
                            padding='SAME',
                            name='conv')
        # conv: [b_s, max_sl, filter_num]
        with tf.variable_scope("global_extractor"):
            # if global_extractor provided, use it as a fake word, do the conv operation
            # with the input x togenther, then send it to the activation function
            if global_infor is not None:
                filter_shape = [1, self.emb_size, self.filter_num]
                conv_g_W = layer.conv_weight_variable(filter_shape, name=level)
                conv_g_b = layer.bias_variable([self.filter_num], name=level)
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
        # output: [b_s, max_seq_len, units]
        concat_vec = []
        self.encode_conv = []
        for i in [3,5,7]:
            conved = self.conv1d(x, i, self.emb_size, level=i)  # (b_s, max_seq_len, filter_num)
            # conved = tf.reduce_max(conved, axis=1)
            concat_vec.append(conved)
            self.encode_conv.append(conved)
        self.encode_conv = tf.stack(self.encode_conv)
        self.encode_conv = tf.transpose(self.encode_conv, [1, 0, 2, 3])
        return tf.concat(concat_vec, -1)

    def d_rnn(self, drnn_input):
        logging.info("##########drnn#########")
        # embedded_words: [batch_size, block, window_size, emb_size]
        drnn_input_reshape = tf.reshape(drnn_input, [-1, self.real_window_size, self.represent_size])
        # drnn_input_reshape: [batch_size * block, window_size, emb_size]
        outputs, _ = tf.nn.dynamic_rnn(cell=self.rnn_cell, dtype=tf.float32, inputs=drnn_input_reshape)
        last_output = outputs[:, -1, :]  # [batch_size * block, 1, rnn_units]
        drnn_output = tf.reshape(last_output, [self.batch_size, -1, self.rnn_units])  # [batch_size, block, rnn_units]
        drnn_output = tf.layers.batch_normalization(drnn_output, training=self.is_train)
        return drnn_output

    def encoder(self, sequence, embedded_words):
        # return: (b_s, max_seq_len, global_encode_units)
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
                logging.info("encode is cnn")
            elif self._config.encode_type == "rnn":
                rnn_encode_cell = tf.nn.rnn_cell.GRUCell(name="encode_gru", num_units=self.rnn_units,
                                                         kernel_initializer=tf.initializers.orthogonal())
                # self.tmp2 = tf.get_variable("rnn/encode_gru/gates/bias:0")
                x = tf.unstack(embedded_words, self.max_seq_len, 1)  # list: [msl, b_s, emd]
                # use static_rnn here, it will raise error if use dynamic_rnn here
                global_encode, last_state = tf.nn.static_rnn(cell=rnn_encode_cell, dtype=tf.float32, inputs=x)
                global_encode = tf.stack(global_encode, 1)
                global_encode_units = self.rnn_units
                logging.info( "*****encode is rnn*******" )
            elif self._config.encode_type == "attend_rnn":
                rnn_encode_cell = tf.nn.rnn_cell.GRUCell(name="encode_gru", num_units=self.rnn_units,
                                                         kernel_initializer=tf.initializers.orthogonal())
                x = tf.unstack(embedded_words, self.max_seq_len, 1)  # list: [msl, b_s, emd]
                global_encode, last_state = tf.nn.static_rnn(cell=rnn_encode_cell, dtype=tf.float32, inputs=x)
                global_encode = tf.stack(global_encode, 1)
                M = tf.nn.tanh(global_encode)
                context_w = layer.conv_weight_variable([1, self.rnn_units])
                context_w = tf.reshape(tf.tile(context_w, [1, self.batch_size]), [self.batch_size, 1, self.rnn_units])
                score = tf.matmul(context_w, M, transpose_b=True)
                score = tf.nn.softmax(tf.squeeze(score))
                self.score = score
                global_encode *= tf.expand_dims(score, -1)
                global_encode_units = self.rnn_units
                logging.info("*****encode is attend_rnn*******" )
            else:
                raise NotImplementedError
            self.global_encode = global_encode
            return global_encode, global_encode_units

    def logits_drnn(self):
        logging.info("##########logit is rnn##########")
        self.represent_size = self.emb_size
        # self.initial_state = self.rnn_cell.zero_state(self.batch_size*self.max_seq_len, dtype=tf.float32)
        # self.initial_state = tf.cast(self.initial_state, tf.float32)

        embedded_words = self.embedding(self.sequence)
        embedded_words = tf.layers.dropout(embedded_words, self.fc_drop, training=self.is_train)
        if self._config.attention_type == "same_init":
            global_encode, global_encode_units = self.encoder(self.sequence, embedded_words)
            if self._config.encode_type == "attend_rnn":
                global_encode = tf.reduce_sum(global_encode, axis=1)
            else:
                global_encode = tf.reduce_max(global_encode, axis=1)
            global_encode_mlp = tf.layers.batch_normalization(global_encode, training=self.is_train)
            global_encode_mlp = layer.fc_fun(
                global_encode_mlp, self.emb_size, initial_type=self._config.initial_type, activation="relu")

            global_encode_mlp = tf.tile(global_encode_mlp, [1, self.max_seq_len])
            global_encode_mlp = tf.reshape(global_encode_mlp, [self.batch_size, self.max_seq_len, self.emb_size])
            self.global_fake = global_encode_mlp
        elif self._config.attention_type == "attend_init":
            global_encode, global_encode_units = self.encoder(self.sequence, embedded_words)
            global_encode_mlp = layer.fc_fun(
                global_encode, self.emb_size, initial_type=self._config.initial_type)
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
        hs = tf.layers.dropout(hs, self.fc_drop, training=self.is_train)

        mlp = tf.matmul(hs, self.W)

        mlp = tf.layers.batch_normalization(mlp, training=self.is_train)
        mlp = tf.nn.relu(mlp)

        fcl_output = tf.matmul(mlp, self.u)
        # self.tmp2 = fcl_output

        return fcl_output

    def cnn(self, x, global_infor):
        logging.info("##########cnn#########")
        self.conv_out = []
        with tf.variable_scope("extractor_cnn"):
            concat_vec = []
            if self._config.attention_type == "attend_init":
                global_infor_abstract = tf.reduce_max(global_infor, axis=1)
                global_infor_abstract = tf.layers.batch_normalization(global_infor_abstract, training=self.is_train)
                global_infor_abstract = layer.fc_fun(global_infor_abstract, self.emb_size, activation="relu")

                global_infor_abstract = tf.reshape(tf.tile(global_infor_abstract, [1, self.max_seq_len]),
                                                   [self.batch_size, self.max_seq_len, -1])
            for filter_i in self.filter_size:
                # filter_shape = [filter_i, self.represent_size, self.filter_num]
                with tf.variable_scope("cnn_filter_{}".format(filter_i)):
                    # conv_W = layer.conv_weight_variable(filter_shape, name=filter_i)
                    # conv_b = layer.bias_variable([self.filter_num], name=filter_i)
                    if self._config.attention_type == "attend_init":
                        pad_input = tf.pad(x, [[0, 0], [filter_i - 1, 0], [0, 0]], mode="CONSTANT")
                        #  print("pad_input:", pad_input.get_shape()) (batch_size, seq_max_len + window_size - 1, embed_size)
                        cnn_blocks = []
                        for tmp_i in range(self.max_seq_len):
                            cnn_blocks.append(tf.slice(pad_input, [0, tmp_i, 0], [-1, filter_i, -1], name='cnn_block'))
                        cnn_blocks = tf.stack(cnn_blocks, 1)  # (batch_size, seq_max_len, filter_i, embed_size)
                        cnn_blocks = tf.reduce_mean(cnn_blocks, 2)

                        global_infor_attend = layer.basic_attention(cnn_blocks, global_infor, name="cnn")
                        global_infor_attend = tf.layers.batch_normalization(global_infor_attend, training=self.is_train)
                        global_infor_attend = layer.fc_fun(global_infor_attend, self.emb_size, activation="relu")

                        global_infor = tf.concat([global_infor_attend, global_infor_abstract], 2)
                        global_infor = tf.layers.batch_normalization(global_infor, training=self.is_train)
                        global_infor = layer.fc_fun(global_infor, self.emb_size,
                                                    initial_type=self._config.initial_type, activation="relu")
                conved = self.conv1d(x, filter_i, self.represent_size, global_infor=global_infor)
                # conved = self.conv1d(x, filter_i, self.represent_size, global_infor=global_infor, conv_W=conv_W, conv_b=conv_b)
                self.conv_out.append(conved)
                conved = tf.reduce_max(conved, axis=1)
                concat_vec.append(conved)
        self.conv_out = tf.stack(self.conv_out)
        self.conv_out = tf.transpose(self.conv_out, [1, 0, 2, 3])
        return tf.concat(concat_vec, -1)

    def logits_cnn_1d(self):
        logging.info("##########logit is cnn##########")
        embedded_words = self.embedding(self.sequence)
        embedded_words = tf.layers.dropout(embedded_words, self.fc_drop, training=self.is_train)
        self.global_fake = None
        self.represent_size = self.emb_size
        if self._config.attention_type == "same_init":
            global_encode, global_encode_units = self.encoder(self.sequence, embedded_words)
            if self._config.encode_type == "attend_rnn":
                global_encode = tf.reduce_sum(global_encode, axis=1)
            else:
                global_encode = tf.reduce_max(global_encode, axis=1)
            global_encode_mlp = tf.layers.batch_normalization(global_encode, training=self.is_train)
            global_encode_mlp = layer.fc_fun(
                global_encode_mlp, self.emb_size, initial_type=self._config.initial_type, activation="relu")
            global_encode_mlp = tf.tile(global_encode_mlp, [1, self.max_seq_len])
            global_encode_mlp = tf.reshape(global_encode_mlp, [self.batch_size, self.max_seq_len, self.emb_size])
            self.global_fake = global_encode_mlp
        elif self._config.attention_type == "attend_init":
            global_encode, global_encode_units = self.encoder(self.sequence, embedded_words)
            # global_encode_mlp = layer.fc_fun(
            #     global_encode, self.emb_size, initial_type=self._config.initial_type)
            # self.global_fake = tf.reshape(global_encode_mlp, [self.batch_size, self.max_seq_len, self.emb_size])
            self.global_fake = global_encode
        else:
            if self._config.attention_type is not None:
                raise NotImplementedError
        outputs = self.cnn(embedded_words, global_infor=self.global_fake)

        with tf.variable_scope("output"):
            outputs = tf.nn.leaky_relu(outputs)
            outputs = tf.layers.batch_normalization(outputs, training=self.is_train)

            fcl_output = layer.fc_fun(outputs, 2000, initial_type='xavier',
                                      activation=self._config.fc_activation_1)
            fcl_output = tf.layers.dropout(fcl_output, rate=self.fc_drop, training=self.is_train)
            fcl_output = layer.fc_fun(fcl_output, self.n_classes, initial_type='xavier')
        return fcl_output

    def logits_cnn_dp(self):
        logging.info("DPCNN")
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
            self.predict_label = tf.argmax(self.logits_op, 1)
        elif self._config.type == 'multi_label':
            loss = tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=label)
            )
        else:
            raise NotImplementedError
        if self._config.l2_loss:
            tv = tf.trainable_variables()
            regularization_cost = tf.add_n([tf.nn.l2_loss(v) for v in tv if 'bias' not in v.name]) * 0.001
            loss += regularization_cost
        return loss

    def loss_encoder(self):
        # train encoder1 firstly then train the encoder1-encoder2 together
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
    window_size = config.model.window_size
    sentence_block_all = []
    for sentence in sentences:
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