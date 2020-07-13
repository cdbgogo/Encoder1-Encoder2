# -*- coding: utf-8 -*-
import logging
import os
import datetime
FORMAT = "%(asctime)s - [%(filename)s:%(lineno)s - %(funcName)5s()] - %(levelname)s - %(message)s"


def get_cur_time():
    time_format = '%m.%d-%H.%M.%S'
    time_stamp = datetime.datetime.now()
    return time_stamp.strftime(time_format)


def logger_init(project_dir, model_name, dataset):
    logger_dir = os.path.join(project_dir, 'log', dataset, model_name)
    if not os.path.exists(logger_dir):
        os.makedirs(logger_dir)

    logger_config = {
        'format': FORMAT,
        'filename': os.path.join(logger_dir, get_cur_time()),
        'level': logging.DEBUG
    }
    logging.basicConfig(**logger_config)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    # tell the handler to use this format
    # console.setFormatter('%(name)-12s: %(levelname)-8s %(message)s')
    # cannot set console formatter
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)
    logging.info("***********logger start***********")
    return logger_config['filename']


if __name__ == '__main__':
    project_dir = './'
    dataset = '1'
    model_name = '2'
    a = logger_init(project_dir, model_name, dataset)
    logger1 = logging.getLogger('myapp.area1')
    logger1.info("123")
