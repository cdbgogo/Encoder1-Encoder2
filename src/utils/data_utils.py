#!/usr/preprocess/env python
#-*- coding: utf-8 -*-

"""
Author	:

Date	:

Brief	: 
"""
""
import numpy as np
import logging
import os
project_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir))
# # print project_dir
# try:
#     from src.model import utils_bert
# except ImportError:
#     import sys
#     sys.path.insert(0, os.path.join(project_dir, 'src', 'model'))
#     import utils_bert

def read_data_bak(path, slot_indexes, slots_lengthes, model_config, delim=';', pad=0, type_dict=None):
    """read_data from disk, format as lego id file
    Args:
        path(string): path
    Returns:
        (Tuple):slots[0], slots[1], ..., slots[n_slots - 1]
    """
    n_slots = len(slot_indexes)
    slots = [[] for _ in range(n_slots)]
    if not type_dict:
        type_dict = {}
    i = 0
    with open(path, 'r') as fin:
        for i, line in enumerate(fin):
            items = line.strip().split(delim)
            try:    
                i += 1
                raw = []
                for index in slot_indexes:
                    slot_value = items[index].split()
                    tp = type_dict.get(index, int)
                    raw.append([tp(x) for x in slot_value])

                for index in range(len(raw)):
                    slots[index].append(pad_and_trunc(raw[index],
                        slots_lengthes[index],
                        pad=pad,
                        sequence=slots_lengthes[index]>1))
            except Exception as e:
                logging.info('%s, Invalid data raw %s' % (e, line.strip()))
                continue
    logging.info("read done, {} lines in total".format(i))
    print(slots[0][:10])
    print(slots[1][:10])
    exit()

    return slots


def read_data(path, n_class, msl, delim=';'):
    """read_data from dataset_for_dev for development
    todo: use the for_dev dataset to do the experiments
    Args:
        path(string): path
    Returns:
        (Tuple):slots[0], slots[1], ..., slots[n_slots - 1]
    """
    # indexes = [0, 1]
    # lengths = [1, self.config.model.max_sequence_length]

    i = 0
    label = []
    sents = []
    with open(path, 'r') as fin:
        for i, line in enumerate(fin):
            items = line.strip().split(delim)
            try:
                i += 1
                # label.append([int(items[0])])
                # logging.info("111111111111")
                # logging.info(i)
                # logging.info(line)

                label_lst = [int(tmp_l) for tmp_l in items[0].split(',')]
                label.append(convert_label(label_lst, n_class))
                tmp_sent = [int(word) for word in items[1].split()]

                mid = msl / 2
                sents.append(tmp_sent[:mid] + tmp_sent[max(mid, len(tmp_sent) - mid):])
                # sents.append(tmp_sent[:msl])

            except Exception as e:
                logging.info(path)
                # logging.info(i)
                logging.info('%s, Invalid data raw %s' % (e, line.strip()))
                continue
    logging.info("read done, {} lines in total\n".format(i))
    return label, sents


def load_vocab(vocab_path):
    vocab_dic = {}
    with open(vocab_path) as f:
        for t_index, line in enumerate(f):
            try:
                line = line.decode('gb18030').encode('utf-8')
                # print(line)
                line = line.strip().split('\t')
                vocab_dic[line[1]] = line[0]
            except:
                print('decode error: {}\t{}'.format(t_index, line))
                vocab_dic[str(t_index)] = "“"
    # print(vocab_dic)
    return vocab_dic


# def transfer_for_t1(vocab_dic, text, msl):
#     text = text.decode('gb18030').encode('utf-8')
#     text = [vocab_dic[t_w] for t_w in text.split()]
#     text = ''.join(text)
#     # print(text)
#     text = text.replace("<unk>", "“")  #  as <unk>
#     # print(text)
#     text = text[:msl+20]
#     tmp_ids = utils_bert.token_for_bert(text)
#     # print(tmp_ids)
#     return tmp_ids
#

# def read_data_for_t1(path, n_class, msl, vocab_word, delim=';'):
#     "tmp fun for t1"
#     i = 0
#     label = []
#     sents = []
#     with open(path, 'r') as fin:
#         for i, line in enumerate(fin):
#             items = line.strip().split(delim)
#             try:
#                 if i % 10000 == 1:
#                     logging.info('read %d lines' % i)
#                 # label.append([int(items[0])])
#                 label_lst = [int(tmp_l) for tmp_l in items[0].split(',')]
#                 label.append(convert_label(label_lst, n_class))
#                 text = items[1]
#                 tmp_ids = transfer_for_t1(vocab_word, text, msl)
#                 tmp_sent = [int(word) for word in tmp_ids]
#                 sents.append(tmp_sent[:msl])
#             except Exception as e:
#                 logging.info('%s, Invalid data raw %s' % (e, line.strip()))
#                 continue
#     logging.info("read t1 done, {} lines in total\n".format(i))
#     return label, sents


def convert_label(label_lst, n_class):
    l = [0] * n_class
    for tmp in label_lst:
        l[tmp] = 1
    return l


def batch_iter(data, config, preprocess_func=None, shuffle=True):
    """
    data: [labels, sents]
    Generates a batch iterator for a dataset.
    """
    batch_size = config.trainer.batch_size
    num_epochs = config.trainer.max_epoch
    labels, sents = data
    # labels = np.array(labels).reshape(-1, 1)
    sents = np.array(sents)
    # print labels
    # exit()
    data_size = len(labels)
    num_batches_per_epoch = int(data_size/batch_size)  # drop the last batch
    # num_batches_per_epoch = int(data_size/batch_size) + 1

    # print('num_epochs: {}, num_batches_per_epoch: {}'.
    #       format(num_epochs, num_batches_per_epoch))
    for epoch in range(num_epochs + 1):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_lables = labels[shuffle_indices]
            shuffled_sents = sents[shuffle_indices]
        else:
            shuffled_lables = labels
            shuffled_sents = sents
        # print(num_batches_per_epoch)
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            if start_index == data_size:  # data_size 100, if [100, 100], continue
                continue
            # print(start_index, end_index)
            sents_batch = shuffled_sents[start_index: end_index]
            labels_batch = shuffled_lables[start_index: end_index]
            # labels_batch = np.asarray(labels_batch)
            # print('sents_batch: {}'.format(sents_batch))
            # print('labels_batch: {}'.format(labels_batch))
            sents_batch_processed, sents_length = process_batch_sents(sents_batch, config, preprocess_func)
            data = [labels_batch, sents_batch_processed, sents_length]
            # print('data: {}'.format(data))
            # exit()
            # print('start_index: {} end_index: {}'.format(start_index, end_index))
            yield epoch+1, batch_num * 100.0 / num_batches_per_epoch, data, []
            # yield epoch, batch_num * 100.0 / num_batches_per_epoch, \
            #         shuffled_data[start_index:end_index]


def batch_iter_new(data, config, epoch, preprocess_func=None, processed_num=0, shuffle=True):
    """
    data: [labels, sents]
    Generates a batch iterator for a dataset.
    """
    batch_size = config.trainer.batch_size
    labels, sents = data
    sents = np.array(sents)
    labels = np.array(labels)
    data_size = len(labels)
    num_batches = int(data_size/batch_size)  # drop the last mini-batch
    # num_batches = int(data_size/batch_size) + 1
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_lables = labels[shuffle_indices]
        shuffled_sents = sents[shuffle_indices]
    else:
        shuffled_lables = labels
        shuffled_sents = sents
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        # print('start_index: {}, end_index: {}, data_size: {}, processed_num {}'.
        #       format(start_index, end_index, data_size, processed_num))
        if start_index == data_size:  # data_size 100, if [100, 100], no size left
            break
        sents_batch = shuffled_sents[start_index: end_index]
        labels_batch = shuffled_lables[start_index: end_index]
        sents_batch_processed, sents_length = process_batch_sents(sents_batch, config, preprocess_func)
        data = [labels_batch, sents_batch_processed, sents_length]
        data_des = (start_index, end_index, data_size, processed_num)
        yield epoch, (end_index + processed_num) * 100.0 / config.data.dataset_size, data, data_des


def process_batch_sents(sents_batch, config, preprocess_func=None, pad=0):
    '''
    pad to same length in a batch to save memory and pad in left
    :param sents_batch: numpy.ndarray
    :param config: whole config
    :param preprocess_func: def in model
    :param pad: sents_batch_processed, sents_length(included pad_in_left)
    :return:
    '''
    # print(sents_batch)
    sents_batch_processed = []
    sents_length = []
    # max_len_in_batch = max([len(sent) for sent in sents_batch])
    # max_len_in_batch = min(max_len_in_batch, config.model.max_sequence_length)
    max_len_in_batch = config.model.max_sequence_length
    # print('max_len_in_batch: {}'.format(max_len_in_batch))
    for sent in sents_batch:
        sent_processed = sent[:]
        for _ in range(config.model.pad_left_num):
            sent_processed = np.insert(sent_processed, 0, pad)
            # sent_processed.insert(0, pad)
        # print('sent_processed: {}'.format(sent_processed))
        sent_processed = sent_processed[: max_len_in_batch]
        sents_length.append(len(sent_processed))
        while len(sent_processed) < max_len_in_batch:
            sent_processed = np.append(sent_processed, pad)
        # print('sent_processed: {}'.format(sent_processed))
        sents_batch_processed.append(sent_processed)
    if preprocess_func:
        sents_batch_processed = preprocess_func(sents_batch_processed, config)
    else:
        sents_batch_processed = np.asarray(sents_batch_processed)
    # print('sents_batch_processed: {}'.format(sents_batch_processed.shape))

    return sents_batch_processed, np.array(sents_length).reshape(-1, 1)


def generate_batches_from_file(train_path, config, preprocess_func=None, samples_in_memory=100000, delim=';'):
    epoch = 1
    logging.info('generate batch from file')
    while epoch <= config.trainer.max_epoch:
        cnt = 0
        processed_num = 0
        logging.info('This is epoch {}'.format(epoch))
        f = open(train_path)
        labels = []
        sents = []
        for real_l, line in enumerate(f):
            items = line.strip().split(delim)
            try:
                # label.append([int(items[0])])
                label_lst = [int(tmp_l) for tmp_l in items[0].split(',')]
                labels.append(convert_label(label_lst, config.model.n_classes))
                sent_tmp = [int(word) for word in items[1].split()]
                sents.append(sent_tmp[:config.model.max_sequence_length])  # to save memory
                cnt += 1
                real_l += 1
                # if real_l % 100000 == 0:
                #     logging.info('read {} lines'.format(real_l))
            except Exception as e:
                logging.info('%s, Invalid data raw %s' % (e, line.strip()))
                continue
            if cnt == samples_in_memory:
                for tmp in batch_iter_new([labels, sents], config, epoch, preprocess_func=preprocess_func,
                                     processed_num=processed_num):
                    yield tmp
                processed_num += cnt
                labels = []
                sents = []
                cnt = 0
        if cnt < samples_in_memory:
            for tmp in batch_iter_new([labels, sents], config, epoch, preprocess_func=preprocess_func,
                                 processed_num=processed_num):
                yield tmp
        f.close()
        epoch += 1


def pad_and_trunc(data, length, pad=0, sequence=False):
    """pad_and_trunc
    Args:
        data(list): .
        length(int): expect length
        pad(int): pad content
    Returns:
        type:
    todo: only need pad to same length in batch, for memory saved
    todo: pad should be decided by model,
        e.g. pad should be 4 if the conv is [3,4,5],
            and be 14 if window_size is 15 in disconnected_rnn
    """
    # Padding in left for sequence
    if pad < 0:
        return data
    if sequence:
        for _ in range(4):
            data.insert(0, pad)

    if len(data) > length:
        return data[:length]

    while len(data) < length:
        data.append(pad)
    return data
