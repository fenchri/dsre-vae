#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 03-Mar-2020

author: fenia
"""

import torch
from torch import nn, optim
import torch.nn.functional as F
from time import time
import datetime
import numpy as np
import os
import copy
from helpers.io import exp_name
torch.set_printoptions(profile="full")


class BaseTrainer:
    def __init__(self, config, device, iterators, vocabs):
        """
        Trainer object.
        Args:
            config (dict): model parameters
            iterators (dict): 'train' and 'test' iterators
        """
        self.config = config
        self.iterators = iterators
        self.vocabs = vocabs
        self.device = device
        self.monitor = {}
        self.best_score = 0
        self.best_epoch = 0
        self.cur_patience = 0
        self.optimizer = None
        self.averaged_params = {}

    @staticmethod
    def print_params2update(main_model):
        print('MODEL:')
        for p_name, p_value in main_model.named_parameters():
            if p_value.requires_grad:
                print('  {} --> Update'.format(p_name))
            else:
                print('  {}'.format(p_name))

    def init_model(self, some_model):
        main_model = some_model(self.config, self.vocabs, self.device)

        # GPU/CPU
        if self.config['device'] != -1:
            torch.cuda.set_device(self.device)
            main_model.to(self.device)
        return main_model

    def set_optimizer(self, main_model):
        optimizer = optim.Adam(main_model.parameters(),
                               lr=self.config['lr'],
                               weight_decay=self.config['weight_decay'],
                               amsgrad=True)
        return optimizer

    @staticmethod
    def _print_start_training():
        print('\n======== START TRAINING: {} ========\n'.format(
            datetime.datetime.now().strftime("%d-%m-%y_%H:%M:%S")))

    @staticmethod
    def _print_end_training():
        print('\n======== END TRAINING: {} ========\n'.format(
            datetime.datetime.now().strftime("%d-%m-%y_%H:%M:%S")))

    def epoch_checking_larger(self, epoch, item):
        """
        Perform early stopping
        Args:
            epoch (int): current training epoch
        Returns (bool): stop or not
        """
        if item > self.best_score:  # improvement
            self.best_score = item
            if self.config['early_stop']:
                self.cur_patience = 0
                self.best_epoch = epoch
            print('Saving checkpoint')
            self.save_checkpoint()
        else:
            self.cur_patience += 1
            if not self.config['early_stop']:
                self.best_epoch = epoch

        if self.config['patience'] == self.cur_patience and self.config['early_stop']:  # early stop must happen
            self.best_epoch = epoch - self.config['patience']
            return True
        else:
            return False

    def save_checkpoint(self):
        torch.save({'model_params': self.model.state_dict(),
                    'vocabs': self.vocabs,
                    'best_epoch': self.best_epoch,
                    'best_score': self.best_score,
                    'optimizer': self.optimizer.state_dict()}, self.save_path)

    def load_checkpoint(self, some_model=None):
        if some_model is not None:
            path = os.path.join(self.config['pretrained_model'], 'bag_re.model')
        else:
            path = os.path.join(self.config['model_folder'], 'bag_re.model')
        checkpoint = torch.load(path)

        self.vocabs = checkpoint['vocabs']
        return checkpoint

    def assign_model(self, checkpoint):
        # Load checkpoint
        model_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['model_params'].items() if
                           (k in model_dict) and (model_dict[k].shape == checkpoint['model_params'][k].shape)}

        print('Loading pre-trained model')
        for d in pretrained_dict.keys():
            print(' ', d)
        print()

        self.model.load_state_dict(pretrained_dict, strict=False)

        self.model.w_vocab = self.vocabs['w_vocab']
        self.model.p_vocab = self.vocabs['p_vocab']
        self.model.r_vocab = self.vocabs['r_vocab']

        # freeze
        if self.config['freeze_pretrained']:
            for p_name, p_value in self.model.named_parameters():
                if p_name in pretrained_dict:
                    p_value.requires_grad = False

        self.model.to(self.device)

    def show_example(self, batch):
        print('\n\n\n')
        print('bag_sizes', torch.sum(batch['bag_size']))
        print(batch['bag_size'])
        print(len(batch['bag_size']))
        print('relation', batch['rel'])
        print(batch['mentions'])
        assert len(batch['source']) == len(batch['target']) == len(batch['mentions']) == len(batch['sent_len'])
        for w1, w2, men, sl, pos1, pos2 in zip(batch['source'], batch['target'], batch['mentions'], batch['sent_len'],
                                               batch['pos1'], batch['pos2']):
            all_w_ids1 = [self.vocabs['w_vocab'].id2word[w_.item()] for w_ in w1]
            all_w_ids2 = [self.vocabs['w_vocab'].id2word[w_.item()] for w_ in w2]
            arg1 = ' '.join(all_w_ids1[men[0]:men[1] + 1])
            arg2 = ' '.join(all_w_ids1[men[2]:men[3] + 1])
            print(arg1, ' ## ', arg2)
            print(' '.join(all_w_ids1))
            print(' '.join(all_w_ids2))
            print([self.vocabs['p_vocab'].id2pos[p1.item()] for p1 in pos1])
            print([self.vocabs['p_vocab'].id2pos[p2.item()] for p2 in pos2])
            print('sentence length', sl)
            assert sum([1 if self.vocabs['w_vocab'].id2word[w_.item()] != '<PAD>' else 0 for w_ in w1]) == sl.item()
            print()
        exit(0)
