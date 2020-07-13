"""
Author	:

Date	:

Brief	: 
"""

import sys
import os.path
import numpy as np
import tensorflow as tf
import datetime
import logging
import copy
import shutil
from distutils.dir_util import copy_tree
# from sklearn import metrics
import json

try:
    from src.utils import data_utils
    from src.model import layer
    from src.utils import config, logger_config
    from src.trainer import Trainer as Trainer_basic
except ImportError:
    from utils import data_utils
    from utils import config, logger_config
    from model import layer
    from trainer import Trainer as Trainer_basic


config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
project_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

# print('base_dir is: {}'.format(base_dir))

tf.set_random_seed(1234)


class Trainer_multilabel(Trainer_basic):
    """Trainer"""

    def evaluate_on_data(self, sess, sequence, labels):
        """evaluate_on_data
        Args:
            sess(type):
            sequence(type):
            labels:
        Returns:
            type:
        """
        # print('pred:\n{}\n'.format(sequence[:10]))
        # print('labels:\n{}\n'.format(labels[:10]))

        logits = []
        loss = 0
        batch_size = self.config.trainer.eval_batch_size
        # print(len(sequence))
        # print(len(labels))

        for i in range(int(len(sequence) / batch_size) + 1):  # to use every data in evaluate, so + 1
            if i * batch_size == len(sequence):  # empty
                break
            t_sequence = sequence[i * batch_size: i * batch_size + batch_size]
            t_label = labels[i * batch_size: i * batch_size + batch_size]
            t_sequence, t_sequence_length = data_utils.process_batch_sents(
                t_sequence, self.config, preprocess_func=self.preprocess_func)
            # print(t_sequence.shape)
            feed_dict = {
                self.model.sequence: t_sequence,
                self.model.label: t_label,
                self.model.sequence_length: t_sequence_length,
                self.model.is_train: False
            }
            t_logits, t_loss = sess.run([self.model.logits_op, self.model.loss_op], feed_dict=feed_dict)
            loss += t_loss
            logits.extend(t_logits)
        # print('sequence len is {}'.format(len(sequence)))
        # print('logits len is {}'.format(len(logits)))
        # print('\n\n')

        # print('logits: {}'.format(logits[:10]))
        # print('labels: {}'.format(labels[:10]))

        loss /= len(sequence)
        # print('true:\n{}\n'.format(true[50:60]))
        # print('logits: {}'.format(logits))
        logits = np.asarray(logits)
        np.savetxt(self.config.model.model_name + '.predict', logits)

        res = []
        res.append(self.evaluate(logits, labels))
        res.append(loss)
        res.append(logits)
        return res

    def evaluate(self, pred, true):
        """evaluate

        Args:
            pred(list): predict result
            true(list): real result
        Returns:
            4-element Tuple: .
        """
        pred = [[1 if label_logit > 0.5 else 0 for label_logit in case_logit] for case_logit in pred]
        A = 0.0
        B = 0.0  # label
        C = 0.0  # pred
        acc = 0.0
        All = 0.0
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        # print('pred is {}'.format(pred))
        # print('true is {}'.format(true))
        for case_label, case_pred in zip(true, pred):
            for logit_l, logit_p in zip(case_label, case_pred):
                All += 1
                if logit_l:
                    B += 1
                if logit_p:
                    C += 1
                if logit_l and logit_p:
                    A += 1
                if logit_p == logit_l:
                    acc += 1
        if A:
            precision = A / C
            recall = A / B
            f1 = 2 * precision * recall / (precision + recall)
        acc /= All
        return acc, precision, recall, f1

    def train(self):
        current_epoch = 1
        eval_delay = 0

        best_test_acc = 0  # to report
        best_test_loss = 100
        best_epoch = 1
        best_epoch_percent = 0
        best_global_step = 0

        best_loss = 100

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
 
        with tf.Session(config=config_tf) as sess:
            sess.run(tf.global_variables_initializer())
            logging.info('Training starts!')
            while True:
                epoch, epoch_percent, batch_slots, data_des = next(self._train_batches)
                # batch_sequence, batch_label = zip(*batch_slots)
                batch_label, batch_sequence, batch_seq_len = batch_slots
                batch_sequence = batch_sequence.astype(np.int32)
                # print('batch_sequence', batch_sequence.shape, type(batch_sequence))
                # print('batch_label', batch_label.shape, type(batch_sequence))
                # print('batch_seq_len', batch_seq_len.shape, type(batch_sequence))
                #
                # print('batch_sequence: {}'.format(batch_sequence[:3]))
                # print('batch_label: {}'.format(batch_label[:3]))
                # print('batch_seq_len: {}'.format(batch_seq_len[:3]))
                # exit()

                feed_dict = {
                    self.model.sequence: batch_sequence,
                    self.model.label: batch_label,
                    self.model.sequence_length: batch_seq_len,
                    self.model.is_train: True,
                }

                # # for debug:
                # fetch_dict_debug = [self.model.tmp]
                # fetch_result = sess.run(fetch_dict_debug, feed_dict=feed_dict)
                # print('last_output: {}'.format(fetch_result[0].shape))
                # exit()

                fetch_dict = [self.train_op,
                              self.model.loss_op,
                              self.global_step]
                # s, d, e, p = data_des
                # logging.info('start_index: {}, end_index: {}, data_size: {}, processed_num: {}'.format(s,d,e,p))

                _, batch_loss, global_step = \
                    sess.run(fetch_dict, feed_dict=feed_dict)
                # print('global_step: {}, epoch: {}'.format(global_step, epoch))
                if global_step and global_step % self.config.show == 0:
                    output_format = 'epoch:{0}[{1:.2f}%] batch_loss:{2}| global_step:{3}'
                    output = [epoch, epoch_percent,
                              batch_loss / self.config.trainer.batch_size, global_step]
                    logging.info(output_format.format(*output))
                    print(output_format.format(*output))

                to_eval = False
                eval_interval = self.config.trainer.eval_batch_interval
                # print(global_step, eval_interval)
                # exit()
                if current_epoch < epoch:
                    # End of an epoch to evaluate
                    to_eval = True
                    logging.info("epoch {} done.".format(current_epoch))
                    print("epoch {} done.".format(current_epoch))
                    format_str = 'epoch {0} finished. Performance on {2}: [acc:{3}, loss:{4}]'

                if self.config.trainer.batch_eval and global_step and global_step % eval_interval == 0:
                    # batch interval to eval
                    to_eval = True
                    format_str = 'epoch {0}, global_step {1}. Performance on {2}: [acc:{3}, loss:{4}]'

                # only_test
                if to_eval:
                    test_acc, test_loss, test_logits = \
                        self.evaluate_on_data(sess, self._sequence_test, self._label_test)
                    # tmp_test_acc = max(tmp_test_acc, test_acc[0])
                    # if test_acc[0] > best_test_acc:
                    if test_loss < best_loss:
                        best_epoch = epoch  # previous: current_epoch, bug when best is in the end of the epoch
                        best_epoch_percent = epoch_percent
                        best_global_step = global_step
                        best_test_acc = test_acc
                        best_test_loss = test_loss
                        best_loss = test_loss
                        eval_delay = 0
                        test_logits = np.asarray(test_logits)
                        np.savetxt(self.config.model.model_name + '.predict', test_logits)
                        self.saver.save(sess, os.path.join(self.checkpoint_dir, 'model.cpkt'), global_step=global_step)
                    else:
                        eval_delay += 1
                    output = [current_epoch, global_step, 'Test', test_acc, test_loss]
                    logging.info(format_str.format(*output))
                    print (format_str.format(*output))

                    epoch_string = 'Best Test Result --- Best Epoch : {} [{:.2f}%] (step is {}, eval delay is {})' \
                                   '---Best test acc: {} ---Best test loss: {}'\
                                   .format(best_epoch,  best_epoch_percent, best_global_step, eval_delay, best_test_acc, best_test_loss)
                    logging.info(epoch_string)
                    print(epoch_string)

                    current_epoch = epoch

                if epoch > self.config.trainer.max_epoch \
                        or eval_delay >= self.config.trainer.max_delay_eval:
                    logging.info("Training Done!")
                    log_time = os.path.basename(self.config.log_file)
                    acc_str = '-' + str(best_test_acc)
                    with open(os.path.join(project_dir, 'log/result'), 'a') as f:
                        f.write("model is {}\nlog_file is: {}\nacc is {}, loss is {}, best epoch is {} [{:.2f}%]\nconfig is: {}\n\n"
                                .format(self.config.model.model_name, self.config.log_file, best_test_acc,
                                        best_test_loss, best_epoch, best_epoch_percent, self.config.log_string))
                    os.rename(self.config.log_file, self.config.log_file + acc_str)

                    # log_time = log_time + acc_str
                    # self.save_model(log_time, acc_str)
                    exit(0)



