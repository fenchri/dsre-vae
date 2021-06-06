#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 03/03/2020

author: fenia
"""

import re
import math
import torch
from torch.utils.data import Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence
import random
import json
import heapq
from tqdm import tqdm
import numpy as np
from .vocabs import *
from collections import OrderedDict


class BagREDataset(Dataset):
    """
    Bag-level relation extraction dataset. Note that relation of NA should be named as 'NA'.
    Code adapted from OpenNRE.
    """
    def __init__(self, path, rel2id, word_vocab, priors, pos_vocab=None, max_sent_length=None,
                 max_vocab=None, max_bag_size=0, mode='train'):
        super().__init__()

        self.rel_vocab = json.load(open(rel2id, 'r')) if rel2id else Relations()
        self.word_vocab = word_vocab if word_vocab else Words()
        self.pos_vocab = pos_vocab if pos_vocab else Positions()

        self.max_bag_size = max_bag_size  # maximum bag size
        self.max_sent_length = max_sent_length
        self.mode = mode
        self.max_vocab = max_vocab
        self.priors = priors
        if priors:
            self.priors_dim = len(self.priors[list(priors.keys())[-1]])

        # Construct bag-level dataset (a bag contains instances sharing the same relation fact)
        self.data = []
        self.bag_scope = {}
        self.name2id, self.id2name = {}, {}
        self.bag_labels, self.bag_offsets = {}, {}
        self.avg_s_len, self.instances = [], 0

        self.unique_sentences = OrderedDict()
        with open(path) as infile:
            for item in tqdm(infile, desc='Loading ' + self.mode.upper()):
                sample = json.loads(item)

                bag_name = sample['bag_name']
                self.name2id[bag_name] = len(self.name2id)
                self.id2name[len(self.name2id) - 1] = bag_name
                self.bag_scope[bag_name] = {'text': [], 'offsets': [], 'labels': sample['bag_labels']}

                for s in sample['sentences']:
                    self.instances += 1
                    txt = s['text']
                    e1 = s['h']['tokens']
                    e2 = s['t']['tokens']
                    tokens_num = len(txt.split(' '))

                    if len(e1) == 1:
                        e1_ = [e1[0]] * tokens_num
                    else:
                        e1_ = [e1[0]] * (e1[0]) + e1 + [e1[-1]] * (tokens_num - e1[-1] - 1)

                    if len(e2) == 1:
                        e2_ = [e2[0]] * tokens_num
                    else:
                        e2_ = [e2[0]] * (e2[0]) + e2 + [e2[-1]] * (tokens_num - e2[-1] - 1)

                    assert len(e1_) == tokens_num and len(e2_) == tokens_num
                    pos1 = np.array(range(tokens_num), 'i') - np.array(e1_)
                    pos2 = np.array(range(tokens_num), 'i') - np.array(e2_)

                    self.unique_sentences[txt] = 1
                    self.avg_s_len += [tokens_num]
                    self.bag_scope[bag_name]['text'] += [txt]
                    self.bag_scope[bag_name]['offsets'] += [{'m1': e1, 'm2': e2, 'pos1': pos1, 'pos2': pos2}]

        print('Unique sentences: ', len(self.unique_sentences))
        if self.mode == 'train':
            self.make_vocabs()

        print("# instances:  {}\n# bags:       {}\n# relations:  {}".format(
              self.instances, len(self.bag_scope), len(self.rel_vocab)))

        # Process data to tensors
        self.make_dataset()
        print()
        
    def make_vocabs(self):
        """
        Construct vocabularies
        """
        # make sure you add unique sentences, to avoid large freq of words due to duplicates
        for us in self.unique_sentences:
            self.word_vocab.add_sentence(us)
        print('Avg. sent length: {:.04}'.format(sum(self.avg_s_len) / len(self.avg_s_len)))

        for pos in range(-self.max_sent_length, self.max_sent_length + 1):
            self.pos_vocab.add_position(pos)

        tot_voc = len(self.word_vocab.word2id)
        if self.max_vocab:
            print('Coverage:         {} %'.format(self.word_vocab.coverage(self.max_vocab)))
            self.word_vocab.resize_vocab_maxsize(self.max_vocab)  # restrict vocab to certain length
        print('Total vocab size: {}/{}'.format(self.word_vocab.n_word, tot_voc))

    def make_dataset(self):
        """
        Convert dataset to tensors
        """
        skipped = 0
        for i in tqdm(range(len(self.bag_scope)), desc='Processing'):
            pair_name = self.id2name[i]
            bag_sents = self.bag_scope[pair_name]['text']
            bag_ent_offsets = self.bag_scope[pair_name]['offsets']
            bag_label = list(set(self.bag_scope[pair_name]['labels']))
            assert len(bag_sents) == len(bag_ent_offsets)

            # binarize labels
            labels = np.zeros((len(self.rel_vocab),), 'i')
            for rel in bag_label:
                labels[self.rel_vocab[rel]] = 1

            bag_seqs, bag_seqs_target, pos1, pos2, sent_len, bag_mentions = [], [], [], [], [], []
            for sentence, mentions in zip(bag_sents, bag_ent_offsets):

                tmp = self.word_vocab.get_ids(sentence, replace=False)
                if self.mode == 'train' or self.mode == 'train-test':
                    tmp = tmp[:self.max_sent_length]  # restrict to max_sent_length

                tmp_source = [self.word_vocab.word2id[self.word_vocab.SOS]] + tmp  # add <SOS>
                tmp_target = tmp + [self.word_vocab.word2id[self.word_vocab.EOS]]  # add <EOS>

                bag_seqs += [torch.tensor(tmp_source).long()]
                bag_seqs_target += [torch.tensor(tmp_target).long()]

                sent_len += [len(tmp_source)]
                bag_mentions += [[mentions['m1'][0] + 1, mentions['m1'][-1] + 1,
                                  mentions['m2'][0] + 1, mentions['m2'][-1] + 1]]

                # Do not forget the additional <PAD> for <SOS>
                pos1_ = [self.pos_vocab.pos2id[self.pos_vocab.PAD]] + \
                         self.pos_vocab.get_ids(mentions['pos1'], self.max_sent_length)
                pos2_ = [self.pos_vocab.pos2id[self.pos_vocab.PAD]] + \
                         self.pos_vocab.get_ids(mentions['pos2'], self.max_sent_length)

                if self.mode == 'train' or self.mode == 'train-test':
                    pos1 += [torch.tensor(pos1_[:self.max_sent_length + 1]).long()]
                    pos2 += [torch.tensor(pos2_[:self.max_sent_length + 1]).long()]
                else:
                    pos1 += [torch.tensor(pos1_).long()]
                    pos2 += [torch.tensor(pos2_).long()]

            if self.priors:
                if pair_name in self.priors:
                    priors = np.asarray(self.priors[pair_name])
                    self.data.append([labels, pair_name + ' ### ' + bag_label[0], len(bag_sents), bag_ent_offsets,
                                      bag_seqs, bag_seqs_target, pos1, pos2, sent_len, bag_mentions, priors])
                else:
                    priors = np.zeros((self.priors_dim,))
                    # if self.mode == 'train' or self.mode == 'train-test':
                    #     skipped += 1
                    #     continue
                    # else:
                    self.data.append([labels, pair_name + ' ### ' + bag_label[0], len(bag_sents), bag_ent_offsets,
                                      bag_seqs, bag_seqs_target, pos1, pos2, sent_len, bag_mentions, priors])
            else:
                priors = None
                self.data.append([labels, pair_name + ' ### ' + bag_label[0], len(bag_sents), bag_ent_offsets,
                                  bag_seqs, bag_seqs_target, pos1, pos2, sent_len, bag_mentions, priors])

        print('Skipped: {}'.format(skipped))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return_list = self.data[index]
        return return_list
