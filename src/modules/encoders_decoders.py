#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 11-Mar-2020

author: fenia
"""

from torch import nn
from .rnn import *
from torch.nn import functional as F


class LSTMEncoder(nn.Module):
    """
    Classic Encoder for Language generation
    """
    def __init__(self, in_features, h_enc_dim, layers_num, dir2, device, action='concat'):
        """
        Args:
            input_dim (int): input dimensionality
            h_enc_dim (int): encoder hidden dimensionality
            layers_num (int): number of hidden layers
            dir2 (bool): bi-directionality
        """
        super(LSTMEncoder, self).__init__()

        self.net = RNNLayer(input_size=in_features,
                            rnn_size=h_enc_dim,
                            num_layers=layers_num,
                            bidirectional=dir2)
        self.hidden = None
        self.dir2 = dir2
        self.h_enc_dim = h_enc_dim
        self.layers_num = layers_num
        self.device = device
        self.action = action

    def init_hidden(self, bs):
        if self.dir2:
            h_0 = torch.zeros(2 * self.layers_num, bs, self.h_enc_dim).to(self.device)
            c_0 = torch.zeros(2 * self.layers_num, bs, self.h_enc_dim).to(self.device)
        else:
            h_0 = torch.zeros(self.layers_num, bs, self.h_enc_dim).to(self.device)
            c_0 = torch.zeros(self.layers_num, bs, self.h_enc_dim).to(self.device)
        return h_0, c_0

    def keep_last_hidden(self, h_state, c_state):
        # layers, directionality, batch_size, dimension
        # keep last layer
        if self.dir2:
            h_state = h_state.view(self.layers_num, 2, -1, self.h_enc_dim)
            h_state = h_state[-1]
            if self.action == 'sum':
                h_state = torch.sum(h_state, dim=0)  # sum
            else:
                h_state = torch.cat(h_state.unbind(dim=0), dim=1)  # concatenation

            c_state = c_state.view(self.layers_num, 2, -1, self.h_enc_dim)
            c_state = c_state[-1]
            if self.action == 'sum':
                c_state = torch.sum(c_state, dim=0)  # sum
            else:
                c_state = torch.cat(c_state.unbind(dim=0), dim=1)  # concatenation
        else:
            h_state = h_state.view(self.layers_num, 1, -1, self.h_enc_dim)
            h_state = h_state[-1].squeeze(dim=0)

            c_state = c_state.view(self.layers_num, 1, -1, self.h_enc_dim)
            c_state = c_state[-1].squeeze(dim=0)
        return h_state, c_state

    def keep_output(self, output):
        if self.dir2:
            output = output.reshape(output.size(1), output.size(0), 2, self.h_enc_dim)

            if self.action == 'sum':
                output = output.sum(dim=2)  # sum
            else:
                output = torch.cat(output.unbind(dim=2), dim=2)  # concatenation

            output = output.view(output.size(1), output.size(0), output.size(2))
        return output

    def forward(self, x, len_=None):
        h_state = self.init_hidden(x.size(0))
        output, (h_state, c_state) = self.net(x, hidden=h_state, lengths=len_)

        h_state, c_state = self.keep_last_hidden(h_state, c_state)
        output = self.keep_output(output)
        return output, (h_state, c_state)


class LSTMDecoder(nn.Module):
    """
    Classic Decoder for language generation
    """
    def __init__(self, in_features, h_dec_dim, layers_num, dir2, device, action='sum'):
        """
        Args:
            input_dim (int): input dimensionality
            h_dec_dim (int): decoder hidden dimensionality
            layers_num (int): number of hidden layers
            dir2 (bool): bi-directionality
        """
        super(LSTMDecoder, self).__init__()

        self.net = RNNLayer(input_size=in_features,
                            rnn_size=h_dec_dim,
                            num_layers=layers_num,
                            bidirectional=dir2)
        self.hidden = None
        self.dir2 = False  # should always be unidirectional
        self.layers_num = layers_num
        self.h_dec_dim = h_dec_dim
        self.device = device
        self.action = action

    def init_hidden(self, bs):
        h_0 = torch.zeros(self.layers_num, bs, self.h_dec_dim).to(self.device)
        c_0 = torch.zeros(self.layers_num, bs, self.h_dec_dim).to(self.device)
        return h_0, c_0

    def keep_last_hidden(self, h_state, c_state):
        # layers, directionality, batch_size, dimension
        # keep last layer
        h_state = h_state.view(self.layers_num, 1, -1, self.h_dec_dim)
        h_state = h_state[-1].squeeze(dim=0)

        c_state = c_state.view(self.layers_num, 1, -1, self.h_dec_dim)
        c_state = c_state[-1].squeeze(dim=0)
        return h_state, c_state

    def forward(self, y, len_=None, hidden_=None):
        if hidden_ is None:
            h_state = self.init_hidden(y.size(0))
        else:
            h_state = hidden_

        output, (h_state, c_state) = self.net(y, hidden=h_state, lengths=len_)  # here h_state is tuple (h_n, c_n)
        return output, (h_state, c_state)
