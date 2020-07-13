#!/usr/preprocess/env python
#-*- coding: utf-8 -*-

"""
Author	:

Date	:

Brief	: Universal Config
"""

import json
import copy
import os
project_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir))


class Config(dict):
    """Config"""
    def __init__(self, config=None, config_file=None):
        # config is the traditional dic converted from json
        # __update is to convert the dic to config.model.filter form
        if config_file:
            with open(config_file, 'r') as fin:
                config = json.load(fin)
        if config:
            self._update(config)

    def add(self, key, value):
        """add

        Args:
                key(type):
                value(type):
        Returns:
                type:
        """
        self.__dict__[key] = value
        
    def _update(self, config):
        if not isinstance(config, dict):
            return

        for key in config:
            if isinstance(config[key], dict):
                config[key] = Config(config[key])

            if isinstance(config[key], list):
                config[key] = [Config(x) if isinstance(x, dict) else x for x in config[key]]
            
        self.__dict__.update(config)

    def __repr__(self):
        return '%s' % self.__dict__


def pre_config(config_file):
    basic_config = os.path.join(project_dir, 'conf', 'basic.config')
    with open(basic_config, 'r') as fin:
        basic_config = json.load(fin)
    with open(config_file, 'r') as fin:
        config = json.load(fin)
    if 'data' in config['data']:
        data_name = config['data']['data']
        config['data'] = basic_config[data_name]

    trainer_name = config['trainer']['trainer']
    config['trainer'] = basic_config[trainer_name]
    config['model']['n_classes'] = config['data']['n_classes']
    config['model']['type'] = config['data']['type']

    # if not specific max_sequence_length in model, use the default in data
    if "max_sequence_length" not in config['model']:
        config['model']['max_sequence_length'] = config['data']['max_sequence_length']
    if 'eval_batch_interval' in config['data']:
        config['trainer']['eval_batch_interval'] = config['data']['eval_batch_interval']
    config['data'].pop("max_sequence_length", None)
    config['data'].pop("eval_batch_interval", None)
    return config


def main():
    """unit test for main"""
    config = Config(config_file='../../conf/cnn_1d/ag_news.model.config')
    a = {
        'trainer': {'a': 1},
        'model': {'haha': 1}
    }
    config = Config(config=a)
    print(config.trainer)
    exit()
    # print(config.log_string)
    trainer_dic = config.__dict__['trainer']

    for arg in vars(config.model):
        print (arg, getattr(config.model, arg))
    exit()
    print(type(trainer_dic))


    print(trainer_dic._update({'rho': 0.8}))
    print(config.trainer)
    # for key, value in config.__dict__.items():
    #     print key, value
    exit()


    print(config.trainer.max_epoch)


if '__main__' == __name__:
    config = pre_config('../../conf/rnn/ag_news.config')
    real_co = Config(config=config)
    real_co.haha = 'haha'
    print(real_co.haha)
    # config.add('haha', 'hahav')
    # print(config.haha)
    # main()

