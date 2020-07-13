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
    import layer
    import modeling
    import utils_tokenization as tokenization
except ImportError:
    from src.model import layer
    from src.model import modeling
    from src.model import utils_tokenization as tokenization

project_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir))

BERT_BASE_DIR=os.path.join(project_dir, 'data', 'chinese_L-12_H-768_A-12')
bert_config_file = os.path.join(BERT_BASE_DIR, 'bert_config.json')
bert_config = modeling.BertConfig.from_json_file(bert_config_file)
vocab_file = os.path.join(BERT_BASE_DIR, 'vocab.txt')
init_checkpoint = os.path.join(BERT_BASE_DIR, 'bert_model.ckpt')


tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)


def token_for_bert(text_a):
    # donot need segments_ids, since in model, the default is all zero
    text_a = text_a.replace(' ', '')
    tokens_a = tokenizer.tokenize(text_a)
    # print tokens_a
    tokens = []
    tokens.append("[CLS]")
    for token in tokens_a:
        tokens.append(token)
    tokens.append("[SEP]")
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = [str(tmp) for tmp in input_ids]
    # print input_ids
    return input_ids


def create_model(input_ids, input_mask, is_training):
    # is_training_python_bool = tf.cond(tf.equal(is_training, tf.constant(True)), lambda: True, lambda: False)
    model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=None,
            use_one_hot_embeddings=False)
    init_bert()
    return model
    output_layer = model.get_pooled_output()

def init_bert():
    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    # if init_checkpoint:
    (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)


if __name__ == '__main__':
    text = "方向盘 向 左 稍 有 偏斜 - - - - 上 四轮定位 调整 两次 OK 。 价格 80 元 ， 4S 要 300 多元 ， 立马 和 他 说"
    text = text.replace(' ', '')
    token_for_bert(text)