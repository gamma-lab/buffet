# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utilities for downloading disc_data from WMT, tokenizing, vocabularies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import tarfile

from six.moves import urllib

from tensorflow.python.platform import gfile
import tensorflow as tf
import numpy as np
import random
from .nltk_dict import tokenizer_nltk

# Special vocabulary symbols - we always put them at the start.
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_unk"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(r"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d")

# URLs for WMT disc_data.
_WMT_ENFR_TRAIN_URL = "http://www.statmt.org/wmt10/training-giga-fren.tar"
_WMT_ENFR_DEV_URL = "http://www.statmt.org/wmt15/dev-v2.tgz"


class Data_Reader:
    def __init__(self, query=None, answer=None, gen=None, train_ratio=0.5, test_ratio=0.5, flag="train"):
        if (flag == "train" and train_ratio == 0.5) or (flag == "test" and test_ratio == 0.5):
            self.data_size = len(query) * 2
        elif flag == "test" and test_ratio != 0.5:
            self.data_size = len(query)
        else:
            raise IOError("Error: Flag should be train or test ")
        self.falg = flag
        self.train_index = 0
        self.shuffle_index_list = self.shuffle_index()
        self.query = query
        self.answer = answer
        self.gen = gen

        self.training_data = self.get_all_training_data()

        if train_ratio == 0.5:
            self.train_ratio = train_ratio
        else:
            raise IOError("Error: ratio should be 0.5 ")

        if test_ratio in [0.0, 0.5, 1.0]:
            self.test_ratio = test_ratio
        else:
            raise IOError("Error: ratio should be from [0.0, 0.5, 1.0] ")

    def get_all_training_data(self):
        training_data = []
        for query_, answer_ in zip(self.query, self.answer):
            training_data.append([query_, answer_, 1])
        for query_, gen_ in zip(self.query, self.gen):
            training_data.append([query_, gen_, 0])
        return training_data

    def get_batch_num(self, batch_size):
        # in each batch, half of the samples are from human while another half from machine.
        x_batch = self.data_size % batch_size
        if x_batch == 0:
            return int(self.data_size / batch_size)
        else:
            return int(self.data_size / batch_size + 1)

    def shuffle_index(self):
        shuffle_index_list = random.sample(range(self.data_size), self.data_size)
        return shuffle_index_list

    def generate_batch_index(self, batch_size):
        if self.train_index + batch_size > self.data_size:
            batch_index = self.shuffle_index_list[self.train_index:self.data_size]
            self.shuffle_index_list = self.shuffle_index()
            remain_size = batch_size - (self.data_size - self.train_index)
            batch_index += self.shuffle_index_list[:remain_size]
            self.train_index = remain_size
        else:
            batch_index = self.shuffle_index_list[self.train_index:self.train_index + batch_size]
            self.train_index += batch_size
        return batch_index

    def generate_training_batch(self, batch_size):
        train_query = []
        train_answer = []
        train_labels = []

        training_index = self.generate_batch_index(batch_size)
        for index in training_index:
            train_query.append(self.training_data[index][0])
            train_answer.append(self.training_data[index][1])
            train_labels.append(self.training_data[index][2])
        return train_query, train_answer, train_labels


    def generate_testing_batch(self, batch_size):
        test_query = []
        test_answer = []
        test_labels = []
        if self.test_ratio == 0.0:
            batch_index = self.generate_batch_index(batch_size)
            for ix, index in enumerate(batch_index):
                test_query.append(self.query[index])
                test_answer.append(self.gen[index])
                if ix % 2 == 0:
                    test_labels.append(0)
                else:
                    test_labels.append(1)
                    # use machine-generated dialogues as both pos and neg samples

        if self.test_ratio == 1.0:
            batch_index = self.generate_batch_index(batch_size)
            for ix, index in enumerate(batch_index):
                test_query.append(self.query[index])
                test_answer.append(self.answer[index])
                if ix % 2 == 0:
                    test_labels.append(0)
                else:
                    test_labels.append(1)
                    # use human-generated dialogues as both pos and neg samples

        if self.test_ratio == 0.5:
            test_batch_index_half_half = self.generate_batch_index(batch_size)
            for index in test_batch_index_half_half:
                test_query.append(self.training_data[index][0])
                test_answer.append(self.training_data[index][1])
                test_labels.append(self.training_data[index][2])
        return test_query, test_answer, test_labels

    def get_batch(self, train_data, bucket_id, batch_size, type=0):

        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs, decoder_inputs = [], []

        # pad them if needed, reverse encoder inputs and add GO to decoder.
        batch_source_encoder, batch_source_decoder = [], []
        # print("bucket_id: %s" %bucket_id)
        if type == 1:
            batch_size = 1
        for batch_i in xrange(batch_size):
            if type == 1:
                encoder_input, decoder_input = train_data[bucket_id]
            elif type == 2:
                # print("disc_data[bucket_id]: ", disc_data[bucket_id][0])
                encoder_input_a, decoder_input = train_data[bucket_id][0]
                encoder_input = encoder_input_a[batch_i]
            elif type == 0:
                encoder_input, decoder_input = random.choice(train_data[bucket_id])
            elif type == 3:
                encoder_input, decoder_input = train_data[batch_i]
                # print("train en: %s, de: %s" %(encoder_input, decoder_input))

            batch_source_encoder.append(encoder_input)
            batch_source_decoder.append(decoder_input)
            # Encoder inputs are padded and then reversed.
            encoder_pad = [PAD_ID] * (encoder_size - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

            # Decoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([GO_ID] + decoder_input +
                                  [PAD_ID] * decoder_pad_size)

        # Now we create batch-major vectors from the disc_data selected above.
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in xrange(encoder_size):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                          for batch_idx in xrange(batch_size)], dtype=np.int32))

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in xrange(decoder_size):
            batch_decoder_inputs.append(
                np.array([decoder_inputs[batch_idx][length_idx]
                          for batch_idx in xrange(batch_size)], dtype=np.int32))

            # Create target_weights to be 0 for targets that are padding.
            batch_weight = np.ones(batch_size, dtype=np.float32)
            for batch_idx in xrange(batch_size):
                # We set weight to 0 if the corresponding target is a PAD symbol.
                # The corresponding target is decoder_input shifted by 1 forward.
                if length_idx < decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == decoder_size - 1 or target == PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)


def maybe_download(directory, filename, url):
    """Download filename from url unless it's already in directory."""
    if not os.path.exists(directory):
        print("Creating directory %s" % directory)
        os.mkdir(directory)
    filepath = os.path.join(directory, filename)
    if not os.path.exists(filepath):
        print("Downloading %s to %s" % (url, filepath))
        filepath, _ = urllib.request.urlretrieve(url, filepath)
        statinfo = os.stat(filepath)
        print("Succesfully downloaded", filename, statinfo.st_size, "bytes")
    return filepath


def gunzip_file(gz_path, new_path):
    """Unzips from gz_path into new_path."""
    print("Unpacking %s to %s" % (gz_path, new_path))
    with gzip.open(gz_path, "rb") as gz_file:
        with open(new_path, "wb") as new_file:
            for line in gz_file:
                new_file.write(line)


def get_wmt_enfr_train_set(directory):
    """Download the WMT en-fr training corpus to directory unless it's there."""
    train_path = os.path.join(directory, "giga-fren.release2.fixed")
    if not (gfile.Exists(train_path + ".fr") and gfile.Exists(train_path + ".en")):
        corpus_file = maybe_download(directory, "training-giga-fren.tar",
                                     _WMT_ENFR_TRAIN_URL)
        print("Extracting tar file %s" % corpus_file)
        with tarfile.open(corpus_file, "r") as corpus_tar:
            corpus_tar.extractall(directory)
        gunzip_file(train_path + ".fr.gz", train_path + ".fr")
        gunzip_file(train_path + ".en.gz", train_path + ".en")
    return train_path


def get_wmt_enfr_dev_set(directory):
    """Download the WMT en-fr training corpus to directory unless it's there."""
    dev_name = "newstest2013"
    dev_path = os.path.join(directory, dev_name)
    if not (gfile.Exists(dev_path + ".fr") and gfile.Exists(dev_path + ".en")):
        dev_file = maybe_download(directory, "dev-v2.tgz", _WMT_ENFR_DEV_URL)
        print("Extracting tgz file %s" % dev_file)
        with tarfile.open(dev_file, "r:gz") as dev_tar:
            fr_dev_file = dev_tar.getmember("dev/" + dev_name + ".fr")
            en_dev_file = dev_tar.getmember("dev/" + dev_name + ".en")
            fr_dev_file.name = dev_name + ".fr"  # Extract without "dev/" prefix.
            en_dev_file.name = dev_name + ".en"
            dev_tar.extract(fr_dev_file, directory)
            dev_tar.extract(en_dev_file, directory)
    return dev_path


def replace_tokens(captions):
    captions = map(lambda x: x.replace('.', ''), captions)
    captions = map(lambda x: x.replace(',', ''), captions)
    captions = map(lambda x: x.replace('"', ''), captions)
    captions = map(lambda x: x.replace('\n', ''), captions)
    captions = map(lambda x: x.replace('?', ''), captions)
    captions = map(lambda x: x.replace('!', ''), captions)
    captions = map(lambda x: x.replace('\\', ''), captions)
    captions = map(lambda x: x.replace('/', ''), captions)


def refine(data):
    data = tokenizer_nltk(data)
    words = re.findall("[a-zA-Z-.?!',]+", data)
    new_word = []
    one_letter=['i', 'I', 'A', 'a']
    two_letter = ['re', 'll', 've']
    for item in words:
        if len(item)==1 and item in one_letter:
            new_word.append(item)
        elif len(item)>=2 and item not in two_letter:
            new_word.append(item)

    # words = ["".join(word.split()) for word in words]
    # words = ["".join(word.split("-")) for word in words]
    data = ' '.join(new_word)
    return data

def filter_tokenizer(sentence):
    line = sentence
    line = re.sub(r'\.+', ' .', line)
    line = re.sub(r'\!+', ' .', line)
    line = re.sub(r'\?+', ' .', line)
    line = re.sub(r'\d+', ' _number_ ', line)
    line = line.replace(".", " . ")
    line = line.replace("!", " . ")
    line = line.replace("?", " ?")
    line = line.replace('"', '')
    line = line.replace('.', ' . ')
    line = line.replace(',', ' , ')
    line = tokenizer_nltk(line)
    line = ' '.join(line.lower().split())
    return line

def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    # sentence = filter_tokenizer(sentence)
    # sentence = refine(sentence)
    for space_separated_fragment in sentence.strip().split():
        if type(space_separated_fragment) == bytes:
            space_separated_fragment = space_separated_fragment.decode()
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
    return [w for w in words if w]


def create_vocabulary(vocabulary_path, data_path_list, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True, threshold=5):
    """Create vocabulary file (if it does not exist yet) from disc_data file.

    Data file is assumed to contain one sentence per line. Each sentence is
    tokenized and digits are normalized (if normalize_digits is set).
    Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
    We write it to vocabulary_path in a one-token-per-line format, so that later
    token in the first line gets id=0, second line gets id=1, and so on.

    Args:
      vocabulary_path: path where the vocabulary will be created.
      data_path: disc_data file that will be used to create vocabulary.
      max_vocabulary_size: limit on the size of the created vocabulary.
      tokenizer: a function to use to tokenize each disc_data sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    print("Creating vocabulary %s from disc_data %s" % (vocabulary_path, data_path_list))
    if gfile.Exists(vocabulary_path):
        print("Found vocabulary file: %s" % vocabulary_path)
    else:
        vocab = {}
        for data_path in data_path_list:
            with gfile.GFile(data_path, mode="r") as f:
                counter = 0
                for line in f:
                    counter += 1
                    if counter % 100000 == 0:
                        print("  processing line %d" % counter)
                    line = tf.compat.as_str_any(line)
                    tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
                    for w in tokens:
                        word = _DIGIT_RE.sub("0", w) if normalize_digits else w
                        if word in vocab:
                            vocab[word] += 1
                        else:
                            vocab[word] = 1

        # sorted_dic = sorted(vocab.values(), reverse=True)
        # word_count_threshold = sorted_dic[threshold]
        # vocab = [w for w in vocab if vocab[w] >= threshold and w != '']
        # print('filtered words from %d to %d' % (len(sorted_dic), len(vocab)))

        # vocab_list = _START_VOCAB + vocab
        # vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        vocab_list = sorted(vocab, key=vocab.get, reverse=True)
        print("Total numbe of tokens: {}".format(len(vocab_list)))
        word_freq=[]
        #for i in range(1000,len(vocab_list), 1000):
        #     word_freq.append([i, vocab[vocab_list[i]]])
        #print("Word Frequence:")
        #print(word_freq)
        final_vocab_list = [] 
        for word in vocab_list:
            if vocab[word]>=max_vocabulary_size:
                final_vocab_list.append(word)   
        final_vocab_list = _START_VOCAB + final_vocab_list
        #if len(vocab_list) > max_vocabulary_size:
        #    vocab_list = vocab_list[:max_vocabulary_size]

        with gfile.GFile(vocabulary_path, mode="w") as vocab_file:
            for w in final_vocab_list:
                vocab_file.write(w + "\n")


def initialize_vocabulary(vocabulary_path):
    """Initialize vocabulary from file.

    We assume the vocabulary is stored one-item-per-line, so a file:
      dog
      cat
    will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
    also return the reversed-vocabulary ["dog", "cat"].

    Args:
      vocabulary_path: path to the file containing the vocabulary.

    Returns:
      a pair: the vocabulary (a dictionary mapping string to integers), and
      the reversed vocabulary (a list, which reverses the vocabulary mapping).

    Raises:
      ValueError: if the provided vocabulary_path does not exist.
    """
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):
    """Convert a string to list of integers representing token-ids.

    For example, a sentence "I have a dog" may become tokenized into
    ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
    "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

    Args:
      sentence: the sentence in bytes format to convert to token-ids.
      vocabulary: a dictionary mapping tokens to integers.
      tokenizer: a function to use to tokenize each sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.

    Returns:
      a list of integers, the token-ids for the sentence.
    """

    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]
    # Normalize digits by 0 before looking words up in the vocabulary.
    return [vocabulary.get(_DIGIT_RE.sub("0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary,
                      tokenizer=None, normalize_digits=True):
    """Tokenize disc_data file and turn into token-ids using given vocabulary file.

    This function loads disc_data line-by-line from data_path, calls the above
    sentence_to_token_ids, and saves the result to target_path. See comment
    for sentence_to_token_ids on the details of token-ids format.

    Args:
      data_path: path to the disc_data file in one-sentence-per-line format.
      target_path: path where the file with token-ids will be created.
      vocabulary_path: path to the vocabulary file.
      tokenizer: a function to use to tokenize each sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    if not gfile.Exists(target_path):
        print("Tokenizing disc_data in %s" % data_path)
        # print("target path: ", target_path)
        # vocab, _ = initialize_vocabulary(vocabulary_path)
        max_id =0
        with gfile.GFile(data_path, mode="r") as data_file:
            with gfile.GFile(target_path, mode="w") as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 100000 == 0:
                        print("  tokenizing line %d" % counter)
                    token_ids = sentence_to_token_ids(line, vocabulary, tokenizer,
                                                      normalize_digits)

                    if len(token_ids)>0:
                        if max(token_ids)>max_id:
                            max_id=max(token_ids)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")
                print("Maximum word ID %d" % max_id)

def prepare_chitchat_data(data_dir, vocabulary, vocabulary_size, tokenizer=None):
    """Get WMT disc_data into data_dir, create vocabularies and tokenize disc_data.

    Args:
      data_dir: directory in which the disc_data sets will be stored.
      en_vocabulary_size: size of the English vocabulary to create and use.
      fr_vocabulary_size: size of the French vocabulary to create and use.
      tokenizer: a function to use to tokenize each disc_data sentence;
        if None, basic_tokenizer will be used.

    Returns:
      A tuple of 6 elements:
        (1) path to the token-ids for English training disc_data-set,
        (2) path to the token-ids for French training disc_data-set,
        (3) path to the token-ids for English development disc_data-set,
        (4) path to the token-ids for French development disc_data-set,
        (5) path to the English vocabulary file,
        (6) path to the French vocabulary file.
    """
    # Get wmt disc_data to the specified directory.
    # train_path = get_wmt_enfr_train_set(data_dir)
    train_path = os.path.join(data_dir, "train")
    # dev_path = get_wmt_enfr_dev_set(data_dir)
    dev_path = os.path.join(data_dir, "dev")
    test_path = os.path.join(data_dir, "test")


    # Create token ids for the training disc_data.
    answer_train_ids_path = train_path + (".ids%d.answer" % vocabulary_size)
    query_train_ids_path = train_path + (".ids%d.query" % vocabulary_size)
    data_to_token_ids(train_path + ".answer", answer_train_ids_path, vocabulary, tokenizer)
    data_to_token_ids(train_path + ".query", query_train_ids_path, vocabulary, tokenizer)

    # Create token ids for the development disc_data.
    answer_dev_ids_path = dev_path + (".ids%d.answer" % vocabulary_size)
    query_dev_ids_path = dev_path + (".ids%d.query" % vocabulary_size)
    data_to_token_ids(dev_path + ".answer", answer_dev_ids_path, vocabulary, tokenizer)
    data_to_token_ids(dev_path + ".query", query_dev_ids_path, vocabulary, tokenizer)

    answer_test_ids_path = test_path + (".ids%d.answer" % vocabulary_size)
    query_test_ids_path = test_path + (".ids%d.query" % vocabulary_size)
    data_to_token_ids(test_path + ".answer", answer_test_ids_path, vocabulary, tokenizer)
    data_to_token_ids(test_path + ".query", query_test_ids_path, vocabulary, tokenizer)

    return (query_train_ids_path, answer_train_ids_path,
            query_dev_ids_path, answer_dev_ids_path,
            query_test_ids_path, answer_test_ids_path)


def prepare_chitchat_data_OpenSub(data_dir, rev_vocab_old=None, vocab_new=None, tokenizer=None):
    train_path = os.path.join(data_dir, "train")
    dev_path = os.path.join(data_dir, "dev")

    # Create token ids for the training disc_data.
    query_train_ids_path = train_path + ".query"
    answer_train_ids_path = train_path + ".answer"
    # gen_train_ids_path = train_path + ".gen"

    split_set_opensub(os.path.join(data_dir, "t_given_s_train.txt"), query_train_ids_path, answer_train_ids_path)
    # split_set_opensub(os.path.join(data_dir, "decoded_train.txt"), query_train_ids_path, gen_train_ids_path,
    #                   rev_vocab_old, vocab_new)

    # Create token ids for the development disc_data.
    query_dev_ids_path = dev_path + ".query"
    answer_dev_ids_path = dev_path + ".answer"
    # gen_dev_ids_path = dev_path + ".gen"

    split_set_opensub(os.path.join(data_dir, "t_given_s_dev.txt"), query_dev_ids_path, answer_dev_ids_path)
    # split_set_opensub(os.path.join(data_dir, "decoded_dev.txt"), query_dev_ids_path, gen_dev_ids_path,
    #                   rev_vocab_old, vocab_new)

    return (query_train_ids_path, answer_train_ids_path, query_dev_ids_path, answer_dev_ids_path)


def hier_prepare_disc_data(data_dir, vocabulary, vocabulary_size, tokenizer=None):
    """Get WMT disc_data into data_dir, create vocabularies and tokenize disc_data.

    Args:
      data_dir: directory in which the disc_data sets will be stored.
      en_vocabulary_size: size of the English vocabulary to create and use.
      fr_vocabulary_size: size of the French vocabulary to create and use.
      tokenizer: a function to use to tokenize each disc_data sentence;
        if None, basic_tokenizer will be used.

    Returns:
      A tuple of 6 elements:

    """
    # Get wmt disc_data to the specified directory.
    # train_path = get_wmt_enfr_train_set(data_dir)
    train_path = os.path.join(data_dir, "train")
    # dev_path = get_wmt_enfr_dev_set(data_dir)
    dev_path = os.path.join(data_dir, "dev")

    # Create token ids for the training disc_data.
    query_train_ids_path = train_path + (".ids%d.query" % vocabulary_size)
    answer_train_ids_path = train_path + (".ids%d.answer" % vocabulary_size)
    gen_train_ids_path = train_path + (".ids%d.gen" % vocabulary_size)

    data_to_token_ids(train_path + ".query", query_train_ids_path, vocabulary, tokenizer)
    data_to_token_ids(train_path + ".answer", answer_train_ids_path, vocabulary, tokenizer)
    data_to_token_ids(train_path + ".gen", gen_train_ids_path, vocabulary, tokenizer)

    # Create token ids for the development disc_data.
    query_dev_ids_path = dev_path + (".ids%d.query" % vocabulary_size)
    answer_dev_ids_path = dev_path + (".ids%d.answer" % vocabulary_size)
    gen_dev_ids_path = dev_path + (".ids%d.gen" % vocabulary_size)

    data_to_token_ids(dev_path + ".query", query_dev_ids_path, vocabulary, tokenizer)
    data_to_token_ids(dev_path + ".answer", answer_dev_ids_path, vocabulary, tokenizer)
    data_to_token_ids(dev_path + ".gen", gen_dev_ids_path, vocabulary, tokenizer)

    return (query_train_ids_path, answer_train_ids_path, gen_train_ids_path,
            query_dev_ids_path, answer_dev_ids_path, gen_dev_ids_path)


def hier_prepare_disc_data_OpenSub(data_dir, rev_vocab=None):
    train_path = os.path.join(data_dir, "train")
    # dev_path = get_wmt_enfr_dev_set(data_dir)
    dev_path = os.path.join(data_dir, "dev")
    test_path = os.path.join(data_dir, "test")

    # Create token ids for the training disc_data.
    query_train_ids_path = train_path + ".query"
    answer_train_ids_path = train_path + ".answer"
    gen_train_ids_path = train_path + ".gen"

    split_set_opensub(os.path.join(data_dir, "t_given_s_train.txt"), query_train_ids_path, answer_train_ids_path)
    split_set_opensub(os.path.join(data_dir, "decoded_train.txt"), query_train_ids_path, gen_train_ids_path)

    # Create token ids for the development disc_data.
    query_dev_ids_path = dev_path + ".query"
    answer_dev_ids_path = dev_path + ".answer"
    gen_dev_ids_path = dev_path + ".gen"

    split_set_opensub(os.path.join(data_dir, "t_given_s_dev.txt"), query_dev_ids_path, answer_dev_ids_path)
    split_set_opensub(os.path.join(data_dir, "decoded_dev.txt"), query_dev_ids_path, gen_dev_ids_path)

    # Create token ids for the training disc_data.
    query_test_ids_path = test_path + ".query"
    answer_test_ids_path = test_path + ".answer"
    gen_test_ids_path = test_path + ".gen"

    split_set_opensub(os.path.join(data_dir, "t_given_s_test.txt"), query_test_ids_path, answer_test_ids_path)
    split_set_opensub(os.path.join(data_dir, "decoded_test.txt"), query_test_ids_path, gen_test_ids_path)

    return (query_train_ids_path, answer_train_ids_path, gen_train_ids_path,
            query_dev_ids_path, answer_dev_ids_path, gen_dev_ids_path,
            query_test_ids_path, answer_test_ids_path, gen_test_ids_path)


def prepare_disc_data(data_dir, vocabulary, vocabulary_size, tokenizer=None):
    train_path = os.path.join(data_dir, "train")
    # dev_path = get_wmt_enfr_dev_set(data_dir)
    dev_path = os.path.join(data_dir, "dev")

    # Create token ids for the training data.
    answer_train_ids_path = train_path + (".ids%d.pos" % vocabulary_size)
    query_train_ids_path = train_path + (".ids%d.neg" % vocabulary_size)
    data_to_token_ids(train_path + ".pos", answer_train_ids_path, vocabulary, tokenizer)
    data_to_token_ids(train_path + ".neg", query_train_ids_path, vocabulary, tokenizer)

    # Create token ids for the development data.
    answer_dev_ids_path = dev_path + (".ids%d.pos" % vocabulary_size)
    query_dev_ids_path = dev_path + (".ids%d.neg" % vocabulary_size)
    data_to_token_ids(dev_path + ".pos", answer_dev_ids_path, vocabulary, tokenizer)
    data_to_token_ids(dev_path + ".neg", query_dev_ids_path, vocabulary, tokenizer)

    return (query_train_ids_path, answer_train_ids_path,
            query_dev_ids_path, answer_dev_ids_path)


def prepare_defined_data(data_path, vocabulary, vocabulary_size, tokenizer=None):
    # vocab_path = os.path.join(data_dir, "vocab%d.all" %vocabulary_size)
    # query_vocab_path = os.path.join(data_dir, "vocab%d.query" %query_vocabulary_size)

    answer_fixed_ids_path = data_path + (".ids%d.answer" % vocabulary_size)
    query_fixed_ids_path = data_path + (".ids%d.query" % vocabulary_size)

    data_to_token_ids(data_path + ".answer", answer_fixed_ids_path, vocabulary, tokenizer)
    data_to_token_ids(data_path + ".query", query_fixed_ids_path, vocabulary, tokenizer)
    return (query_fixed_ids_path, answer_fixed_ids_path)


def get_dummy_set(dummy_path, vocabulary, vocabulary_size, tokenizer=None):
    dummy_ids_path = dummy_path + (".ids%d" % vocabulary_size)
    data_to_token_ids(dummy_path, dummy_ids_path, vocabulary, tokenizer)
    dummy_set = []
    with gfile.GFile(dummy_ids_path, "r") as dummy_file:
        line = dummy_file.readline()
        counter = 0
        while line:
            counter += 1
            dummy_set.append([int(x) for x in line.split()])
            line = dummy_file.readline()
    return dummy_set


def vocab_transfer(id_1, rev_vocab_old, vocab_new):
    # training set
    word = rev_vocab_old.get(int(id_1), '_UNK')
    new_id = vocab_new.get(word, UNK_ID)
    return new_id


def split_set_opensub(data_path, source_path, target_path):
    if not gfile.Exists(source_path):
        print("Tokenizing disc_data in %s" % data_path)
        # print("target path: ", target_path)
        # vocab, _ = initialize_vocabulary(vocabulary_path)
        with gfile.GFile(data_path, mode="r") as data_file:
            if not gfile.Exists(source_path):
                with gfile.GFile(source_path, mode="w") as tokens_file_s:
                    with gfile.GFile(target_path, mode="w") as tokens_file_t:
                        for line in data_file:
                            source, target = line.strip().split('|')
                            source = source.strip().split()
                            target = target.strip().split()
                            tokens_file_s.write(
                                " ".join([str(int(tok) + 2) for tok in source]) + "\n")
                            tokens_file_t.write(
                                " ".join([str(int(tok) + 2) for tok in target]) + "\n")
    elif not gfile.Exists(target_path):
        with gfile.GFile(data_path, mode="r") as data_file:
            with gfile.GFile(target_path, mode="w") as tokens_file_t:
                for line in data_file:
                    source, target = line.strip().split('|')
                    target = target.strip().split()
                    tokens_file_t.write(
                        " ".join([str(int(tok) + 2) for tok in target]) + "\n")
