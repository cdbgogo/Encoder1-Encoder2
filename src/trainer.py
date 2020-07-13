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
import json

try:
    from src.utils import data_utils
    from src.model import layer
    from src.model import optimization
    from src.utils import config, logger_config
except ImportError:
    from utils import data_utils, config, logger_config
    from model import layer
    from model import optimization


config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
project_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

# print('base_dir is: {}'.format(base_dir))

tf.set_random_seed(1234)


class Trainer(object):
    """Trainer"""
    def __init__(self, model, config, preprocess_func=None):
        self.model = model
        self.preprocess_func = preprocess_func
        self.config = config  # whole config
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.current_epoch = 1

        # self.global_step = tf.train.get_or_create_global_step()
        if config.model.model_name == 'bert':
            logging.info('make_train_op for bert')
            self.train_op = self.make_train_op_bert()
        elif config.trainer.optimizer == 'Adamdecay':
            self.train_op = self.make_train_op_adam_decay()
        elif config.trainer.lr_decay:
            # self.train_op, self.is_warmup = self.make_train_op_decay_lr()
            self.train_op = self.make_train_op_decay_lr()
        else:
            self.train_op = self.make_train_op()
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.checkpoint_dir = os.path.join(base_dir, self.config.trainer.checkpoint_dir_base,
                                           self.config.data.dataset, self.config.model.model_name)
        # if os.path.exists(self.checkpoint_dir):  # delete model before, all model will be saved in model_saved/model_bak
        #     shutil.rmtree(self.checkpoint_dir)
        self.config.show = 400  # show batch_loss how many batch step

    def load_saver_and_predict(self):
        """load_saver_and_predict"""
        self.checkpoint_dir = self.config.trainer.checkpoint_dir
        checkpoint_file = tf.train.latest_checkpoint(self.checkpoint_dir)
        with tf.Session(config=config_tf) as sess:
            self.saver.restore(sess, checkpoint_file)
            self.predict_on_data(sess, self._sequence_test, self._label_test)

    def load_data_debug(self, debug_setting):
        """load_data for debug"""
        data_path = os.path.join(base_dir, self.config.data.test_data)
        self._label_test, self._sequence_test = data_utils.read_data(data_path, self.config.model.n_classes,
                                                                     self.config.model.max_sequence_length)
        self.config.trainer.batch_size = debug_setting['batch_size']
        self.config.trainer.max_epoch = debug_setting['max_epoch']
        self.config.trainer.eval_batch_interval = debug_setting['eval_batch_interval']
        test_num = debug_setting['test_number']
        self.config.show = debug_setting['show']

        if debug_setting['mode'] == 'run':
            self._label_test, self._sequence_test = self._label_test[:test_num], self._sequence_test[:test_num]
            labels, sequence_train = copy.deepcopy(self._label_test), copy.deepcopy(self._sequence_test)
            self._label_dev, self._sequence_dev = copy.deepcopy(self._label_test), copy.deepcopy(self._sequence_test)
        elif debug_setting['mode'] == 'same':
            one_data_label, one_data_seq = self._label_test[:1], self._sequence_test[:1]
            self._label_test, self._sequence_test = one_data_label*test_num, one_data_seq*test_num
            labels, sequence_train = copy.deepcopy(self._label_test), copy.deepcopy(self._sequence_test)
            self._label_dev, self._sequence_dev = copy.deepcopy(self._label_test), copy.deepcopy(self._sequence_test)
        self._train_batches = data_utils.batch_iter([labels, sequence_train], config=self.config,
                                                    preprocess_func=self.preprocess_func, shuffle=False)

    def load_data(self):
        data_path = os.path.join(base_dir, self.config.data.test_data)
        logging.info("Load test data ... from {}".format(data_path))
        if self.config.model.model_name == 'bert' and self.config.data.dataset == 'topic_l1':
            vocab_word = data_utils.load_vocab(self.config.data.vocab_path)
            self._label_test, self._sequence_test = data_utils.read_data_for_t1(data_path, self.config.model.n_classes,
                                                                         self.config.model.max_sequence_length, vocab_word)
            data_path = os.path.join(base_dir, self.config.data.train_data)
            logging.info("Load train data ... from {}".format(data_path))
            # labels, sequence_train = data_utils.read_data(data_path, self.config.model.n_classes)
            self._train_batches = data_utils.generate_batches_from_file_for_t1(data_path, self.config, vocab_word,
                                                                        preprocess_func=self.preprocess_func,
                                                                        samples_in_memory=300000)
            return

        self._label_test, self._sequence_test = data_utils.read_data(data_path, self.config.model.n_classes,
                                                                     self.config.model.max_sequence_length)
        data_path = os.path.join(base_dir, self.config.data.dev_data)
        logging.info("Load dev data ... from {}".format(data_path))
        self._label_dev, self._sequence_dev = data_utils.read_data(data_path, self.config.model.n_classes,
                                                                         self.config.model.max_sequence_length)
        data_path = os.path.join(base_dir, self.config.data.train_data)
        logging.info("Load train data ... from {}".format(data_path))
        # labels, sequence_train = data_utils.read_data(data_path, self.config.model.n_classes)
        if not self.config.predict_mode:
            self._train_batches = data_utils.generate_batches_from_file(data_path, self.config,
                                                                        preprocess_func=self.preprocess_func,
                                                                        samples_in_memory=300000)

    def evaluate_on_data(self, sess, sequence, labels, istest=False):
        logits = []
        loss = []
        evaluate_data_size = len(sequence)
        batch_size = self.config.trainer.eval_batch_size
        for i in range(int(len(sequence) / batch_size) + 1):  # to use every data in evaluate, so + 1
            if i * batch_size == len(sequence):  # empty
                break
            t_sequence = sequence[i * batch_size: i * batch_size + batch_size]
            t_label = labels[i * batch_size: i * batch_size + batch_size]
            if len(t_sequence) != batch_size:
                t_sequence.extend(sequence[:batch_size])  # fake test data
                t_label.extend(labels[:batch_size])  # fake test data
                t_sequence = t_sequence[:batch_size]
                t_label = t_label[:batch_size]
            assert len(t_sequence) == len(t_label) == batch_size
            t_sequence, t_sequence_length = data_utils.process_batch_sents(
                t_sequence, self.config, preprocess_func=self.preprocess_func)

            feed_dict = {
                self.model.sequence: t_sequence,
                self.model.label: t_label,
                self.model.sequence_length: t_sequence_length,
                self.model.is_train: False
            }
            t_logits, t_loss = sess.run([self.model.logits_op, self.model.loss_op], feed_dict=feed_dict)
            # t_logits, t_loss, train_symbol_show = sess.run([self.model.logits_op, self.model.loss_op, self.model.train_symbol_show], feed_dict=feed_dict)
            # print train_symbol_show
            # exit()
            loss.append(t_loss)
            logits.extend(t_logits)
        # loss = loss[:evaluate_data_size]
        logits = logits[:evaluate_data_size]
        loss = np.sum(loss) / evaluate_data_size
        # loss is sum of a batch, so the loss is not equal t0 the real loss since the fake input in last batch

        if istest:
            np.savetxt(self.config.model.model_name + '.predict', logits)
        pred = list(map(np.argmax, logits))
        true = list(map(np.argmax, labels))
        res = []
        res.append(self.evaluate(pred, true))
        res.append(loss)
        res.append(logits)
        return res

    def predict_on_data(self, sess, sequence, labels):
        logits = []
        scores = []
        filter_out = []
        predicts = []
        loss = []
        evaluate_data_size = len(sequence)
        batch_size = self.config.trainer.eval_batch_size
        for i in range(int(len(sequence) / batch_size) + 1):  # to use every data in evaluate, so + 1
            if i * batch_size == len(sequence):  # empty
                break
            t_sequence = sequence[i * batch_size: i * batch_size + batch_size]
            t_label = labels[i * batch_size: i * batch_size + batch_size]
            if len(t_sequence) != batch_size:
                t_sequence.extend(sequence[:batch_size])  # fake test data
                t_label.extend(labels[:batch_size])  # fake test data
                t_sequence = t_sequence[:batch_size]
                t_label = t_label[:batch_size]
            assert len(t_sequence) == len(t_label) == batch_size
            t_sequence, t_sequence_length = data_utils.process_batch_sents(
                t_sequence, self.config, preprocess_func=self.preprocess_func)
            feed_dict = {
                self.model.sequence: t_sequence,
                self.model.label: t_label,
                self.model.sequence_length: t_sequence_length,
                self.model.is_train: False
            }
            if self.config.model.attention_type is not None and self.config.model.encode_type == "cnn":
                t_logits, t_loss, t_score, t_fliter_mo, t_predict = sess.run(
                    [self.model.logits_op, self.model.loss_op, self.model.encode_conv,
                     self.model.conv_out, self.model.predict_label], feed_dict=feed_dict)
                scores.extend(t_score)
            else:
                t_logits, t_loss, t_fliter_mo, t_predict = sess.run(
                    [self.model.logits_op, self.model.loss_op,
                     self.model.conv_out, self.model.predict_label], feed_dict=feed_dict)
            filter_out.extend(t_fliter_mo)
            predicts.extend(t_predict)
            loss.append(t_loss)
            logits.extend(t_logits)
        filter_out = np.asarray(filter_out)[:evaluate_data_size]
        predicts = np.asarray(predicts)[:evaluate_data_size]
        vocab_path = os.path.join(project_dir, self.config.data.vocab_path)
        acc = 0.0
        labels = np.argmax(labels, axis=1)
        for p, t in zip(predicts, labels):
            if p == t:
                acc += 1
        acc /= evaluate_data_size
        vocab = {}
        for line in open(vocab_path):
            items = line.strip().split('\t')
            vocab[items[0]] = items[1]
        sequence_w = []
        for tmp_s in sequence:
            tmp_s = [vocab[str(tmp_w)] for tmp_w in tmp_s]
            sequence_w.append(tmp_s)
        # to_save = [sequence_w, logits, filter_out, scores, predicts, labels]
        # print filter_out.shape  # (num, 3, msl, 128)
        # exit()
        filter_out = np.transpose(filter_out, [0, 2, 1, 3])  # (num, msl, 3, 128)
        filter_out = np.maximum(filter_out, 0)
        # filter_out = np.square(filter_out)
        # filter_out = np.sqrt(np.sum(filter_out, axis=3))
        filter_out = np.sum(filter_out, axis=3)
        filter_out = np.sum(filter_out, axis=2)
        filter_out = filter_out.astype(dtype=np.float16)
        # print filter_out[0]

        if self.config.model.attention_type is not None and self.config.model.encode_type == "cnn":
            scores = np.asarray(scores)[:evaluate_data_size]
            scores = np.transpose(scores, [0, 2, 1, 3])  # (num, msl, 3, 128)
            scores = np.maximum(scores, 0)
            # scores = np.square(scores)
            # scores = np.sqrt(np.sum(scores, axis=3))
            scores = np.sum(scores, axis=3)
            scores = np.sum(scores, axis=2)
            scores = scores.astype(dtype=np.float16)
            to_save = [filter_out, scores]
        else:
            to_save = [filter_out]
        to_save = np.asarray(to_save)
        to_save_2 = [predicts, labels]
        to_save_2 = np.asarray(to_save_2)
        name = "{}.{}-{}.{}.{:.3}.{}".format(self.config.data.dataset, self.config.model.encode_type,
                                          self.config.model.feature_type, self.config.model.attention_type,
                                             acc, "result")
        saving_dir = os.path.join(project_dir, "output")
        if not os.path.exists(saving_dir):
            os.makedirs(saving_dir)
        import pickle as pkl
        with open(os.path.join(saving_dir, name+".seq"), "w") as f:
            pkl.dump(sequence_w, f)
        with open(os.path.join(saving_dir, name), "w") as f:
            logging.info("saving to {}".format(name))
            np.savez_compressed(f, score=to_save, label=to_save_2)

    def evaluate(self, pred, true):
        """evaluate

        Args:
            pred(list): predict result
            true(list): real result
        Returns:
            4-element Tuple: .
        """
        TP = np.float(0.0)
        for p, t in zip(pred, true):
            if p == t:
                TP += 1
        # TP = len(filter(lambda (p, t): p == t, zip(pred, true))) * np.float(1.0)
        acc = TP / len(pred)
        return acc

    def make_train_op(self):
        """make_train_op"""
        optimizer = layer.get_optimizer(self.config.trainer.optimizer, self.config.trainer)
        # for batch_normalization
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        tvars = tf.trainable_variables()
        with tf.control_dependencies(extra_update_ops):
            gradients = optimizer.compute_gradients(self.model.loss_op, tvars)
            # capped_gvs = [(tf.clip_by_value(grad, 1e-8, 1), var) for grad, var in gradients]
            #  don't clip grad in (1e-8, 1), it is usually in clip_y
            if self.config.trainer.debug:
                # capped_gvs = gradients
                for grad, var in gradients:
                    print var, grad
                capped_gvs = [(tf.clip_by_norm(grad, 5.0), var) for grad, var in gradients]
                # capped_gvs = [(tf.clip_by_value(grad, -1, 1), var) for grad, var in gradients]
            elif self.config.trainer.clip_value:
                logging.info("clip grad by value")
                capped_gvs = [(tf.clip_by_value(grad, -1, 1), var) for grad, var in gradients]
            else:
                logging.info("clip grad by norm")
                capped_gvs = [(tf.clip_by_norm(grad, 5.0), var) for grad, var in gradients if grad is not None]
            # capped_gvs = gradients
            train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)
        return train_op

    def train(self):
        """train"""

        eval_delay = 0

        best_test_acc = 0  # to report
        best_test_loss = 100
        best_epoch = 1
        best_epoch_percent = 0
        best_global_step = 0

        tmp_test_loss = 100  # tmp best loss
        tmp_test_acc = 0.0  # tmp best acc
        tmp_best_epoch_step = 0

        best_dev_acc = 0.0
        best_dev_loss = 1000.0

        epoch_train_all_correct = 0.0
        epoch_train_all_loss = 0.0
        epoch_train_pass_sample = 0.0

        intervel_train_all_correct = 0.0
        intervel_train_all_loss = 0.0
        intervel_train_pass_sample = 0.0

        test_acc = 0
        test_loss = 100

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
 
        with tf.Session(config=config_tf) as sess:
            sess.run(tf.global_variables_initializer())
            logging.info('Training starts!')
            while True:
                to_eval = False
                eval_interval = self.config.trainer.eval_batch_interval

                epoch, epoch_percent, batch_slots, data_des = next(self._train_batches)
                batch_label, batch_sequence, batch_seq_len = batch_slots
                batch_sequence = batch_sequence.astype(np.int32)

                # print('batch_sequence', batch_sequence.shape)
                # print('batch_label', batch_label.shape)
                # print('batch_seq_len', batch_seq_len.shape)

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
                # fetch_dict_debug = [self.model.tmp1, self.model.tmp2]
                # fetch_result = sess.run(fetch_dict_debug, feed_dict=feed_dict)
                # print('tmp1: {}'.format(fetch_result[0].shape))
                # # print(fetch_result[0][0])
                # print(np.mean(fetch_result[0][0]))
                # print("****************\n\n")
                #
                # print('tmp2: {}'.format(fetch_result[1].shape))
                # # print(fetch_result[1][0])
                # print(np.mean(fetch_result[1][0]))
                # exit()

                fetch_dict = [self.train_op,
                              self.model.loss_op,
                              self.global_step,
                              self.model.result_op]

                if self.current_epoch < epoch:
                    # End of an epoch to evaluate
                    if self.config.trainer.epoch_eval:
                        to_eval = True
                    epoch_train_summary = "\n***---Epoch train summary! epoch {} done, seen {} samples. epoch average loss:{}\n epoch average precisoin {}---***\n".\
                        format(self.current_epoch, epoch_train_pass_sample, epoch_train_all_loss/epoch_train_pass_sample, epoch_train_all_correct/epoch_train_pass_sample)
                    logging.info(epoch_train_summary)
                    epoch_train_all_loss, epoch_train_all_correct, epoch_train_pass_sample = 0.0, 0.0, 0.0
                    format_str = 'epoch {0} finished. Performance on {2}: [acc:{3}, loss:{4}]'
                    self.current_epoch = epoch

                # _, batch_loss, global_step, is_warm, correct_prediction = \
                #     sess.run(fetch_dict, feed_dict=feed_dict)
                _, batch_loss, global_step, correct_prediction = \
                    sess.run(fetch_dict, feed_dict=feed_dict)
                # print("global step: {}".format(global_step_))

                # print global_step
                # print train_symbol_show

                # # v = sess.graph.get_tensor_by_name("encode_module/rnn/encode_gru/candidate/kernel:0")
                # v = sess.graph.get_tensor_by_name("encode_module/rnn/encode_gru/gates/kernel:0")
                # t = sess.graph.get_tensor_by_name("rnn/gru_cell/gates/kernel:0")
                # # m = sess.graph.get_tensor_by_name("gradients/dense/MatMul_grad/tuple/control_dependency:0")
                # fetch_dict_debug = [v, t]
                # fetch_result = sess.run(fetch_dict_debug, feed_dict=feed_dict)
                # # print sum(fetch_result[0])
                # # print v
                # print "v---", np.sum(fetch_result[0])
                # print "t---", np.sum(fetch_result[1])
                # # print "m---", np.sum(fetch_result[2])

                epoch_train_all_correct += correct_prediction
                epoch_train_pass_sample += batch_sequence.shape[0]
                epoch_train_all_loss += batch_loss

                intervel_train_all_correct += correct_prediction
                intervel_train_pass_sample += batch_sequence.shape[0]
                intervel_train_all_loss += batch_loss

                if global_step and global_step % self.config.show == 0:
                    output_format = 'epoch:{0}[{1:.2f}%] batch_loss:{2}| global_step:{3}'
                    output = [epoch, epoch_percent,
                              batch_loss / batch_sequence.shape[0], global_step]
                    logging.info(output_format.format(*output))

                if self.config.trainer.batch_eval and global_step and global_step % eval_interval == 0:
                    # batch interval to eval
                    to_eval = True
                    format_str = 'epoch {0}, global_step {1}. Performance on {2}: [acc:{3}, loss:{4}]'
                # test dev
                if to_eval:
                    # Test on dev
                    intervel_train_summary = "\n----Train intervel summary! This intervel seen {} samples. intervel average loss:{}\n intervel average precisoin {}---\n". \
                        format(intervel_train_pass_sample, intervel_train_all_loss / intervel_train_pass_sample,
                               intervel_train_all_correct / intervel_train_pass_sample)
                    logging.info(intervel_train_summary)

                    dev_acc, dev_loss, dev_logits = \
                        self.evaluate_on_data(sess, self._sequence_dev, self._label_dev)
                    output = [self.current_epoch, global_step, 'Dev', dev_acc, dev_loss]
                    logging.info(format_str.format(*output))

                    # test_acc, test_loss, test_logits = \
                    #     self.evaluate_on_data(sess, self._sequence_test, self._label_test)
                    # tmp_test_acc = max(tmp_test_acc, test_acc)
                    # if dev_acc > best_dev_acc or best_dev_loss > dev_loss:
                    if dev_acc >= best_dev_acc:
                        # if dev_acc > best_dev_acc:
                        best_dev_acc = dev_acc
                        # else:
                        #     best_dev_loss = dev_loss
                        test_acc, test_loss, test_logits = \
                            self.evaluate_on_data(sess, self._sequence_test, self._label_test, istest=False)
                        tmp_test_acc = max(tmp_test_acc, test_acc)

                        best_epoch = epoch
                        best_epoch_percent = epoch_percent
                        best_global_step = global_step
                        best_test_acc = test_acc
                        # self.predict_on_data(sess, self._sequence_test, self._label_test)
                        if best_test_acc >= tmp_test_acc:
                            tmp_test_acc = best_test_acc
                            tmp_best_epoch_percent = 'epoch {} [{:.2f}%]'.format(epoch, epoch_percent)
                        eval_delay = 0
                        self.saver.save(sess, os.path.join(self.checkpoint_dir, 'model.cpkt'), global_step=global_step)
                    else:
                        eval_delay += 1

                    output = [self.current_epoch, global_step, 'Test', test_acc, test_loss]
                    logging.info(format_str.format(*output))

                    # print (intervel_train_all_loss, intervel_train_all_correct, intervel_train_pass_sample)
                    intervel_train_all_loss, intervel_train_all_correct, intervel_train_pass_sample = 0.0, 0.0, 0.0

                    intervel_string = 'Best Test Result --- Best Epoch : {} [{:.2f}%] (step is {}, eval delay is {})' \
                                   '---Best test acc: {:4f}---tmp_best_acc is {:4f} {})' \
                                   '---Best dev acc: {:4f}---Best dev loss: {:4f}\n' \
                                   .format(best_epoch, best_epoch_percent, best_global_step, eval_delay,
                                           best_test_acc, tmp_test_acc, tmp_best_epoch_percent,
                                           best_dev_acc, best_dev_loss)

                    # epoch_string = 'Best Test Result --- Best Epoch : {} [{:.2f}%] (step is {}, eval delay is {})' \
                    #                '---Best test acc: {} ---Best test loss: {} ---tmp test loss {}'\
                    #                .format(best_epoch, best_epoch_percent, best_global_step, eval_delay,
                    #                        best_test_acc, best_test_loss, tmp_test_loss)
                    logging.info(intervel_string)


                #
                # # only_test
                # if to_eval:
                #     # print('to_eval')
                #     # Test on dev
                #
                #     test_acc, test_loss, test_logits = \
                #         self.evaluate_on_data(sess, self._sequence_test, self._label_test)
                #     # tmp_test_acc = max(tmp_test_acc, test_acc)
                #     if test_acc > best_test_acc:
                #         best_epoch = epoch  # previous: current_epoch, bug when best is in the end of the epoch
                #         best_epoch_percent = epoch_percent
                #         best_global_step = global_step
                #         best_test_acc = test_acc
                #         best_test_loss = test_loss
                #         eval_delay = 0
                #         test_logits = np.asarray(test_logits)
                #         np.savetxt(self.config.model.model_name + '.predict', test_logits)
                #         self.saver.save(sess, os.path.join(self.checkpoint_dir, 'model.cpkt'), global_step=global_step)
                #     else:
                #         eval_delay += 1
                #     if tmp_test_loss > test_loss:
                #         tmp_test_loss = test_loss
                #     output = [current_epoch, global_step, 'Test', test_acc, test_loss]
                #     logging.info(format_str.format(*output))
                #     print (format_str.format(*output))
                #
                #     epoch_string = 'Best Test Result --- Best Epoch : {} [{:.2f}%] (step is {}, eval delay is {})' \
                #                    '---Best test acc: {} ---Best test loss: {} ---tmp test loss {}'\
                #                    .format(best_epoch, best_epoch_percent, best_global_step, eval_delay,
                #                            best_test_acc, best_test_loss, tmp_test_loss)
                #     logging.info(epoch_string)
                #     print(epoch_string)
                #
                #     current_epoch = epoch

                if epoch > self.config.trainer.max_epoch \
                        or eval_delay >= self.config.trainer.max_delay_eval:
                    logging.info("Training Done!")
                    # log_time = os.path.basename(self.config.log_file)
                    # acc_str = '-' + str(best_test_acc)
                    # with open(os.path.join(project_dir, 'log/result'), 'a') as f:
                    #     f.write("model is {}\nlog_file is: {}\nacc is {}, loss is {}, best epoch is {} [{:.2f}%]\nconfig is: {}\n\n"
                    #             .format(self.config.model.model_name, self.config.log_file, best_test_acc,
                    #                     best_test_loss, best_epoch, best_epoch_percent, self.config.log_string))
                    # os.rename(self.config.log_file, self.config.log_file + acc_str)

                    # log_time = log_time + acc_str
                    # self.save_model(log_time, acc_str)
                    exit(0)

    def save_model(self, log_time, acc_str):
        self.model_bak_dir = os.path.join(base_dir, self.config.trainer.checkpoint_dir_base, 'model_bak',
                                          self.config.data.dataset, self.config.model.model_name, log_time)
        if not os.path.exists(self.model_bak_dir):
            os.makedirs(self.model_bak_dir)
        shutil.copy(self.config.log_file + acc_str, self.model_bak_dir)
        shutil.copy(self.config.config_file, self.model_bak_dir)
        copy_tree(self.checkpoint_dir, self.model_bak_dir)

    def make_train_op_bert(self):
        """make_train_op"""
        num_train_epochs = 3
        warmup_proportion = 0.1
        num_train_steps = int(self.config.data.dataset_size / self.config.trainer.batch_size * num_train_epochs)
        num_warmup_steps = int(num_train_steps * warmup_proportion)
        bert_recommend_lr = 5e-5
        train_op = optimization.create_optimizer(
            self.model.loss_op, self.config.trainer.learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)
        return train_op

    def make_train_op_adam_decay(self):
        """make_train_op"""
        logging.info ("use Adamdecay")

        num_train_epochs = 10
        warmup_proportion = 0.1
        num_train_steps = int(self.config.data.dataset_size / self.config.trainer.batch_size * num_train_epochs)
        num_warmup_steps = int(num_train_steps * warmup_proportion)
        train_op = optimization.create_optimizer_adam_decay(
            self.model.loss_op, self.config.trainer.learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)
        return train_op

    def make_train_op_decay_lr(self):
        """make_train_op"""
        logging.info("use lr decay")
        encoder_fixed_epoch = self.config.trainer.encoder_fixed_epoch
        num_train_epochs = 15.0
        warmup_proportion = encoder_fixed_epoch / num_train_epochs
        num_train_steps = int(self.config.data.dataset_size / self.config.trainer.batch_size * num_train_epochs)
        num_warmup_steps = int(num_train_steps * warmup_proportion)
        if self.config.model.attention_type is not None and encoder_fixed_epoch:
            loss_encode = self.model.loss_encode_op
        else:
            loss_encode = None
        train_op, is_warm = optimization.create_optimizer_new(self.model.loss_op, self.config.trainer,
                                                     num_train_steps, num_warmup_steps, loss_encode=loss_encode)
        return train_op
