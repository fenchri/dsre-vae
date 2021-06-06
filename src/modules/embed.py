#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 03-Mar-2020
author: fenia
"""

import torch
import numpy as np
from torch import nn, torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EmbedLayer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, ignore=None, freeze=False, pretrained=None,
                 mapping=None):
        """
        Args:
            num_embeddings (tensor): number of unique items
            embedding_dim (int): dimensionality of vectors
            pretrained (str): pretrained embeddings
            mapping (dict): mapping of items to unique ids
        """
        super(EmbedLayer, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.freeze = freeze

        self.embedding = nn.Embedding(num_embeddings=num_embeddings,
                                      embedding_dim=embedding_dim,
                                      padding_idx=ignore)

        if pretrained is not None:
            print('Initialising with pre-trained word embeddings!')
            self.init_pretrained(pretrained, mapping)
        self.embedding.weight.requires_grad = not freeze

    def init_pretrained(self, pretrained, mapping):
        """
        Args:
            pretrained (dict): keys are words, values are vectors
            mapping (dict): keys are words, values are unique ids
        Returns: updates the embedding matrix with pre-trained embeddings
        """
        found = 0
        for word in mapping.keys():  # words in vocabulary
            if word in pretrained:
                self.embedding.weight.data[mapping[word], :] = torch.from_numpy(pretrained[word])
                found += 1
            elif word.lower() in pretrained:
                self.embedding.weight.data[mapping[word], :] = torch.from_numpy(pretrained[word.lower()])
                found += 1

        print('Assigned {:.2f}% words a pre-trained word embedding\n'.format(found * 100 / len(mapping)))
        assert (self.embedding.weight[mapping['.']].to('cpu').data.numpy() == pretrained['.']).all(), \
            'ERROR: Embeddings not assigned'

    def forward(self, xs):
        """
        Args:
            xs (tensor): [batchsize, word_ids]
        Returns (tensor): [batchsize, word_ids, dimensionality]
        """
        embeds = self.embedding(xs)
        return embeds
