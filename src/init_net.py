#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 10/09/2020

author: fenia
"""

import torch
from torch import nn, torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from modules.embed import *
from modules.encoders_decoders import *
from modules.attention import *
from modules.ModifiedAdaptiveSoftmax import AdaptiveLogSoftmaxWithLoss


class BaseNet(nn.Module):
    """
    Model architecture
    """
    def __init__(self, config, vocabs, device):
        """
        Args:
            params (dict): model parameters
            vocab (class): class structure with word vocabulary
            device (int): gpu or cpu (-1) device
        """
        super(BaseNet, self).__init__()

        self.in_drop = nn.Dropout(p=config['input_dropout'])
        self.out_drop = nn.Dropout(p=config['output_dropout'])
        self.device = device
        self.config = config

        self.PAD_id = vocabs['w_vocab'].word2id[vocabs['w_vocab'].PAD]
        self.EOS_id = vocabs['w_vocab'].word2id[vocabs['w_vocab'].EOS]
        self.SOS_id = vocabs['w_vocab'].word2id[vocabs['w_vocab'].SOS]
        self.UNK_id = vocabs['w_vocab'].word2id[vocabs['w_vocab'].UNK]

        if config['include_positions']:
            input_dim = config['word_embed_dim'] + 2*config['pos_embed_dim']
        else:
            input_dim = config['word_embed_dim']

        self.w_embed = EmbedLayer(num_embeddings=vocabs['w_vocab'].n_word,
                                  embedding_dim=config['word_embed_dim'],
                                  pretrained=vocabs['w_vocab'].pretrained,
                                  ignore=vocabs['w_vocab'].word2id[vocabs['w_vocab'].PAD],
                                  mapping=vocabs['w_vocab'].word2id,
                                  freeze=config['freeze_words'])

        self.r_embed = EmbedLayer(num_embeddings=len(vocabs['r_vocab']),
                                  embedding_dim=config['rel_embed_dim'])

        self.p_embed = EmbedLayer(num_embeddings=vocabs['p_vocab'].n_pos,
                                  embedding_dim=config['pos_embed_dim'],
                                  ignore=vocabs['p_vocab'].pos2id[vocabs['p_vocab'].PAD])

        self.lang_encoder = LSTMEncoder(in_features=input_dim,
                                        h_enc_dim=config['enc_dim'],
                                        layers_num=config['enc_layers'],
                                        dir2=config['enc_bidirectional'],
                                        device=self.device,
                                        action='sum')

        self.sentence_attention = SelectiveAttention(device=self.device)
        self.dim2rel = nn.Linear(in_features=config['rel_embed_dim'], out_features=len(vocabs['r_vocab']))
        self.dim2rel.weight = self.r_embed.embedding.weight  # tie weights

        # task loss
        self.task_loss = nn.BCEWithLogitsLoss(reduction='none')

        if self.config['reconstruction']:
            self.hid2mu = nn.Linear(2*config['enc_dim'], config['latent_dim'])  # produce vectors!
            self.hid2var = nn.Linear(2*config['enc_dim'], config['latent_dim'])
            self.latent2hid = nn.Linear(config['latent_dim'], config['dec_dim'])

            self.reduction = nn.Linear(in_features=config['latent_dim'] + 2 * config['enc_dim'],
                                       out_features=config['rel_embed_dim'],
                                       bias=False)

            decoder_dim = config['word_embed_dim'] + config['latent_dim']
            self.lang_decoder = LSTMDecoder(in_features=decoder_dim,
                                            h_dec_dim=config['dec_dim'],
                                            layers_num=config['dec_layers'],
                                            dir2=config['dec_bidirectional'],
                                            device=self.device,
                                            action='sum')

            # Modified version with loss 'none' instead of 'mean' (and make my own)
            # remove PAD and SOS ids (-2): we don't need to predict those
            self.reco_loss = AdaptiveLogSoftmaxWithLoss(config['dec_dim'], vocabs['w_vocab'].n_word,
                                                        cutoffs=[round(vocabs['w_vocab'].n_word/15),
                                                                 3*round(vocabs['w_vocab'].n_word/15)])
        else:
            self.reduction = nn.Linear(in_features=3 * config['enc_dim'],
                                       out_features=config['rel_embed_dim'],
                                       bias=False)
