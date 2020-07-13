# coding:utf-8


import sys
import csv
import nltk
import re
import numpy as np
import os
import argparse
import random
import sys
import codecs
import random

#  fix print unicodedecode bug
sys.stdout = codecs.getwriter('utf8')(sys.stdout)
sys.stderr = codecs.getwriter('utf8')(sys.stderr)
project_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
random.seed(1996)


class Preprocess():
    def __init__(self, debug):
        self.SYMBOL_P = re.compile('([-/.:*~+=,`]|\\\\)')
        self.word_dict = {}
        self.word_count = {}
        self.token_lines = []
        self.emb_dict = {}
        self.data_dir = data_dir
        self.debug = debug

    def clean_str(self, string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = string.strip().strip('"')
        string = re.sub(r"[^A-Za-z0-9(),!?\.\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"\.", " \. ", string)
        string = re.sub(r"\"", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def conv_input(self, words):
        words = [str(self.word_dict.get(w, 1)) for w in words]
        return ' '.join(words)

    def tokenize(self, line):
        line = self.SYMBOL_P.sub(r' \g<1> ', line)
        words = nltk.tokenize.word_tokenize(line.decode('utf-8'))
        # in glove, embedding of Find and find both exist, so we follow baoxin's way to preprocess

        # line = self.clean_str(line.decode('utf-8'))  # use clean_str
        # return line.split()
        return words

    def add_random(self):
        random_embedding = 0.01 * np.random.uniform(-1, 1, 300)
        return list(random_embedding)

    def build_embedding(self, emb_path, token_lines):
        in_pretrained = 0
        emb_word_set = set()
        with open(emb_path) as f:
            for line in f:
                line = line.strip()
                items = line.split()
                if len(items) < 10:
                    continue
                emb_word_set.add(items[0])
                if items[0] in self.word_dict:
                    self.emb_dict[items[0]] = np.array(map(float, items[1:]))
                    in_pretrained += 1
        # print(self.emb_dict)  #  {"are": array}
        print("embedding in pretrained is {}, words is {}".format(in_pretrained, len(self.word_dict)))
        context_dict = {}
        for words, label in token_lines:
            for i, w in enumerate(words):
                if w not in emb_word_set and w in self.word_dict:
                    if w not in context_dict:
                        context_dict[w] = []
                    context_dict[w].extend(words[max(0, i - 4):i + 5])

        for word in context_dict:
            words = context_dict[word]
            words = [w for w in words[1:] if w in self.emb_dict]
            if words:
                word_array = np.array(map(lambda x: self.emb_dict[x], words))
                self.emb_dict[word] = word_array.mean(0)

        # print(self.emb_dict.keys())
        # exit()

    def build_dict(self, n_words=50000):
        print("all words is {}".format(len(self.word_count)))
        word_items = self.word_count.items()
        word_items.sort(key=lambda x: x[1], reverse=True)
        for index, (word, count) in enumerate(word_items):
            if index % 2500 == 0:
                try:
                    print("{}\t{}\t{}".format(index, word.encode("utf-8"), count))
                except:
                    pass

        with open(os.path.join(data_dir, "unigram.id"), 'w') as f:
            f.write("{}\t{}\t{}\n".format(0, "<pad>", ""))
            f.write("{}\t{}\t{}\n".format(1, "<unk>", ""))
            self.word_dict["<pad>"] = 0
            self.word_dict["<unk>"] = 1

            for index, (word, count) in enumerate(word_items):
                index_ = index + 2
                self.word_dict[word] = index_
                # print self.word_dict
                f.write("{}\t{}\t{}\n".format(index_, word.encode('utf-8'), count))
                if "sogou" in self.path:
                    if count < 4:
                        break
                else:
                    if index_ >= n_words-1:
                        break
        # self.word_set = set(zip(*(word_items[:n_words]))[0])
        # print(self.word_dict)

    def load_csv(self, path, train=False):
        self.token_lines = []
        self.path = path
        with open(path, 'rb') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                s = ' '.join(row[1:])
                words = self.tokenize(s)
                if train:
                    for word in set(words):  # words count only one in a sentence
                        if word not in self.word_count:
                            self.word_count[word] = 0
                        self.word_count[word] += 1
                self.token_lines.append((words, row[0]))
                if self.debug:
                    break

    def write_emb(self, path):
        embed = []
        word_dic_sort = sorted(self.word_dict.items(), key=lambda x: x[1])
        for word, _ in word_dic_sort:
            # print word,self.emb_dict[word]
            if word in self.emb_dict:
                embed.append(list(self.emb_dict[word]))
            else:
                embed.append(self.add_random())
        embed = np.asarray(embed)
        embed[0] = np.zeros(300, dtype='float32')
        np.savetxt(path, embed)

    def write_data(self, path, token_lines):
        with open(path, 'w') as f:
            for words, label in token_lines:
                f.write("{};{}\n".format(int(label)-1, self.conv_input(words)))


def split_train_and_val():
    with open(train_path) as f:
        train_lines = f.readlines()

    with open(test_path) as f:
        test_lines = f.readlines()

    random.shuffle(train_lines)
    n_test = len(test_lines)
    dir_path = os.path.dirname(train_path)
    with open(os.path.join(dir_path, train_split_path), 'w') as f:
        for line in train_lines[n_test:]:
            f.write(line)
    with open(os.path.join(dir_path, val_split_path), 'w') as f:
        for line in train_lines[:n_test]:
            f.write(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help="data path")
    parser.add_argument("-n", "--n_words", type=int, help="n_words")
    parser.add_argument("-d", '--debug', action='store_true',
                        help="if --debug, debug mode")
    args = parser.parse_args()
    if args.path is None or args.n_words is None:
        print("-p indicates the data_path; -n indicates the n_words")
        exit()

    data_dir = args.path
    n_words = args.n_words
    data_dir = os.path.join(project_dir, data_dir)
    train_path = os.path.join(data_dir, 'train.csv')
    train_split_path = os.path.join(data_dir, 'train_train.csv')
    val_split_path = os.path.join(data_dir, 'train_dev.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    split_train_and_val()

    p = Preprocess(debug=args.debug)
    p.load_csv(train_split_path, train=True)
    print('building dict')
    p.build_dict(n_words=int(n_words))

    # print('building embedding')
    # embedding_path = os.path.join(project_dir, 'data', 'raw_data', 'glove.840B.300d.txt')
    # p.build_embedding(embedding_path, p.token_lines)

    # p.write_emb(os.path.join(dir_path, 'glove.' + n_words + '.embedding'))
    # p.write_emb(os.path.join(data_dir, 'neighbor_True.embedding'))

    print('preprocessing dataset')
    p.write_data(train_split_path + '.id', p.token_lines)
    p.load_csv(val_split_path, train=False)
    p.write_data(val_split_path + '.id', p.token_lines)
    p.load_csv(test_path, train=False)
    p.write_data(test_path + '.id', p.token_lines)

    train_all_path = os.path.join(os.path.dirname(train_split_path), 'train.csv.id')
    # print(train_all_path)
    with open(train_split_path + '.id') as f_1, open(val_split_path + '.id') as f_2, \
            open(train_all_path, 'w') as f_3:
        lines1 = f_1.readlines()
        lines2 = f_2.readlines()
        lines1.extend(lines2)
        for line in lines1:
            f_3.write(line)

    model_data_dir = data_dir.replace("raw_data", "model_data")
    if not os.path.exists(model_data_dir):
        os.makedirs(model_data_dir)
    os.system("cp {}/train*.id {}".format(data_dir, model_data_dir))
    os.system("cp {}/test.csv.id {}".format(data_dir, model_data_dir))
    os.system("cp {}/unigram.id {}".format(data_dir, model_data_dir))
    os.system("cp {}/neighbor_True.embedding {}".format(data_dir, model_data_dir))

