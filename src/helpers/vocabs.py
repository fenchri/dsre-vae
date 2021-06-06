#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 03/03/2020

author: fenia
"""

import re
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random
import six
from collections import OrderedDict


def load_pretrained_embeds(pret_embeds_file, embedding_dim):
    pretrained = {}
    with open(pret_embeds_file, 'r') as infile:
        for line in infile:
            line = line.rstrip().split(' ')
            word, vec = line[0], list(map(float, line[1:]))
            if (word not in pretrained) and (len(vec) == embedding_dim):
                pretrained[word] = np.asarray(vec, 'f')
    print('Loaded {}, {}-dimensional pretrained word-embeddings\n'.format(len(pretrained), embedding_dim))
    return pretrained


class Positions(object):
    def __init__(self):
        self.PAD = '<PAD>'

        self.pos2id = {self.PAD: 0}
        self.id2pos = {0: self.PAD}
        self.n_pos = 1
        self.pos2count = {self.PAD: 0}

    def add_position(self, position):
        if position not in self.pos2id:
            self.pos2id[position] = self.n_pos
            self.pos2count[position] = 1
            self.id2pos[self.n_pos] = position
            self.n_pos += 1
        else:
            self.pos2count[position] += 1

    def mapping(self, x, max_pos):
        if -max_pos <= x <= max_pos:
            return self.pos2id[x]
        elif x < -max_pos:
            return self.pos2id[-max_pos]
        elif x > max_pos:
            return self.pos2id[max_pos]
        else:
            print('error!')
            print(x, max_pos)
            exit(0)

    def get_ids(self, pos, max_pos):
        pos = list(map(lambda x: self.mapping(x, max_pos), pos))
        return pos


class Relations(object):
    def __init__(self):
        self.NONE = 'NA'

        self.rel2id = {self.NONE: 0}
        self.id2rel = {0: self.NONE}
        self.n_rel = 1
        self.rel2count = {self.NONE: 0}
        self.facts = {}

    def add_relation(self, relation):
        if relation not in self.rel2id:
            self.rel2id[relation] = self.n_rel
            self.rel2count[relation] = 1
            self.id2rel[self.n_rel] = relation
            self.n_rel += 1
        else:
            self.rel2count[relation] += 1


class Words(object):
    """
    The Vocab Class, holds the vocabulary of a corpus and
    mappings from tokens to indices and vice versa.
    """
    def __init__(self):
        self.PAD = '<PAD>'
        self.SOS = '<SOS>'
        self.EOS = '<EOS>'
        self.UNK = '<UNK>'

        self.word2id = {self.PAD: 0, self.SOS: 1, self.UNK: 2, self.EOS: 3}
        self.id2word = {0: self.PAD, 1: self.SOS, 2: self.UNK, 3: self.EOS}
        self.n_word = 4
        self.word2count = {}
        self.min_freq = 1
        self.pretrained = None

    def add_word(self, word):
        if word not in self.word2id:
            self.word2id[word] = self.n_word
            self.word2count[word] = 1
            self.id2word[self.n_word] = word
            self.n_word += 1
        else:
            self.word2count[word] += 1

    def add_sentence(self, sentence):
        # sentence is assumed already tokenized
        for word in sentence.split(' '):
            self.add_word(word)

    def coverage(self, top_n):
        high2low_freq = {k: v for k, v in sorted(self.word2count.items(), key=lambda item: item[1], reverse=True)}
        occurrences = [freq for tok, freq in high2low_freq.items()]
        total = sum(occurrences)
        cov = sum(occurrences[:top_n]) / total
        return np.round(cov, 4) * 100

    def get_tokens(self, id_keys):
        return [self.id2word[key] for key in id_keys]

    def get_ids(self, words, replace=False):
        word_keys = words.split(' ')
        if replace:
            ids = []
            for key in word_keys:
                if (self.word2count[key] == 1) and (random.uniform(0, 1) < float(0.5)):
                    ids += [self.word2id[self.UNK]]
                else:
                    ids += [self.word2id[key]]
        else:
            ids = [self.word2id[key] if key in self.word2id else self.word2id[self.UNK] for key in word_keys]
        return ids

    def resize_vocab_maxsize(self, max_vocab_size):
        """
        Reform vocabulary based on a maximum size
        """
        high2low_freq = OrderedDict(
            {k: v for k, v in sorted(self.word2count.items(), key=lambda item: item[1], reverse=True)})
        keep_words = list(high2low_freq.keys())[:max_vocab_size]   # keep most frequent (high2low)

        self.word2id = {self.PAD: 0, self.SOS: 1, self.UNK: 2, self.EOS: 3}
        self.word2id.update({w: v+4 for v, w in enumerate(keep_words)})
        self.word2count = {w: high2low_freq[w] for w in keep_words}
        self.n_word = max_vocab_size+4
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.min_freq = high2low_freq[keep_words[-1]]
        print('min_freq:        ', self.min_freq)
        print('max_freq:        ', high2low_freq[keep_words[0]])
