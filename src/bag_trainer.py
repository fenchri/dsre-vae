#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 03-Mar-2020

author: fenia
"""

import torch
from torch import nn, optim
import torch.nn.functional as F
from bag_net import BagNet as target_model
from modules.trainer import BaseTrainer
from helpers.io import *
from helpers.metrics import *
from time import time
import numpy as np
import json
torch.set_printoptions(profile="full")


class Trainer(BaseTrainer):
    def __init__(self, config, device, iterators=None, vocabs=None):
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

        self.model = None  # self.init_model(target_model)
        self.optimizer = None   # self.set_optimizer(self.model, opt=self.config['optimizer'])

        self.primary_metric = []  # on the validation set
        self.best_score = 0
        self.best_epoch = 0
        self.cur_patience = 0
        self.averaged_params = {}
        self.iterations = len(self.iterators['train'])
        self.save_path = os.path.join(self.config['model_folder'], 'bag_re.model')

    def optimise(self):
        """
        Main Training Loop.
        """
        self.print_params2update(self.model)
        print_options(self.config)
        self._print_start_training()

        if self.config['priors']:
            kl_w = self.kl_anneal_function('constant', self.config['epochs'] * self.iterations)
        else:
            kl_w = self.kl_anneal_function('logistic', self.config['epochs'] * self.iterations)
        for epoch in range(1, self.config['epochs'] + 1):
            train_tracker, time_ = self.train_epoch(epoch, kl_w)
            _, _, _ = self.calculate_performance(epoch, train_tracker, time_, mode='train')

            val_tracker, time_ = self.eval_epoch(iter_name='val')
            val_perf, _, _ = self.calculate_performance(epoch, val_tracker, time_, mode='val')
            self.primary_metric += [val_perf[self.config['primary_metric']]]

            stop = self.epoch_checking_larger(epoch, self.primary_metric[-1])
            if stop:
                break
            print()

        self._print_end_training()
        return self.best_score

    def calculate_performance(self, epoch, tracker, time_, mode='train'):
        performance, p_points, r_points = calc_all_perf(
            y_true=np.vstack(tracker['gtruth']),
            y_scores=np.vstack(tracker['probabilities']),
            nclasses=len(self.vocabs['r_vocab']), mode=mode
        )

        for item in ['total', 'task', 'reco', 'kld', 'ppl']:
            tracker[item] = np.mean(tracker[item])

        print_performance(epoch, tracker, performance, time_, name=mode)
        return performance, p_points, r_points

    @staticmethod
    def init_tracker():
        return {'total': [], 'reco': [], 'kld': [], 'kld_w': [], 'task': [], 'ppl': [],
                'probabilities': [], 'gtruth': [], 'total_bags': 0, 'total_sents': 0}

    @staticmethod
    def kl_anneal_function(anneal_function, steps, k=0.0025, x0=2500):
        """ Credits to: https://github.com/timbmg/Sentence-VAE/blob/master/train.py#L63 """
        if anneal_function == 'logistic':
            return [float(1 / (1 + np.exp(-k * (step - x0)))) for step in range(steps)]
        elif anneal_function == 'linear':
            return [min(1, int(step / x0)) for step in range(steps)]
        else:
            return [1] * steps

    def run(self):
        """
        Main Training Loop.
        """
        self.print_params2update(self.model)
        print_options(self.config)
        self._print_start_training()

        if self.config['priors']:
            kl_w = self.kl_anneal_function('constant', self.config['epochs'] * self.iterations)
        else:
            kl_w = self.kl_anneal_function('logistic', self.config['epochs'] * self.iterations)

        for epoch in range(1, self.config['epochs'] + 1):
            train_tracker, time_ = self.train_epoch(epoch, kl_w)

            _, _, _ = self.calculate_performance(epoch, train_tracker, time_, mode='train')

            val_tracker, time_ = self.eval_epoch(iter_name='val')
            val_perf, _, _ = self.calculate_performance(epoch, val_tracker, time_, mode='val')

            self.primary_metric += [val_perf[self.config['primary_metric']]]

            stop = self.epoch_checking_larger(epoch, self.primary_metric[-1])
            print('current best epoch:', self.best_epoch)
            if stop:
                break
            print()

        self._print_end_training()

        print('Best epoch: {}\n'.format(self.best_epoch))
        return self.best_score

    def train_epoch(self, epoch, kl_w):
        """
        Evaluate the model on the train set.
        """
        t1 = time()
        self.model = self.model.train()
        tracker = self.init_tracker()

        iterations = len(self.iterators['train'])
        for batch_idx, batch in enumerate(self.iterators['train']):
            step = ((epoch-1) * iterations) + batch_idx

            tracker['total_bags'] += batch['rel'].size(0)
            tracker['total_sents'] += torch.sum(batch['bag_size'], dim=0).item()

            for keys in batch.keys():
                if keys != 'bag_names':
                    batch[keys] = batch[keys].to(self.device)

            if self.config['show_example']:
                self.show_example(batch)

            self.model.zero_grad()
            task_loss, rel_probs, kld, rec_loss, latent_z = self.model(batch)  # forward pass

            if self.config['reconstruction']:
                loss = (self.config['task_weight'] * task_loss) + \
                       (1 - self.config['task_weight']) * (rec_loss['sum'] + (kl_w[step] * kld))
            else:
                loss = task_loss

            loss.backward()

            # gradient clipping
            if self.config['clip'] > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config['clip'])
            self.optimizer.step()   # update

            # collect logits & losses
            tracker['probabilities'] += [rel_probs.cpu().data.numpy()]
            tracker['gtruth'] += [batch['rel'].cpu().data.numpy()]
            tracker['total'] += [loss.item()]
            tracker['reco'] += [rec_loss['sum'].item()]
            tracker['ppl'] += [np.exp(rec_loss['mean'].item())]  # mean loss for PPL
            tracker['kld'] += [kld.item()]
            tracker['task'] += [task_loss.item()]
            tracker['kld_w'] += [kl_w[step]]

            if batch_idx % self.config['log_interval'] == 0:
                print('Step {:<6}    LOSS = {:10.4f}    TASK = {:10.4f}    RECO = {:10.4f}    '
                      'KL = {:10.6f}    KL_W = {:.04f}    PPL = {:10.4f}'.format(
                       step,
                       loss.item(), tracker['task'][-1], tracker['reco'][-1],
                       tracker['kld'][-1], tracker['kld_w'][-1], tracker['ppl'][-1]))

        t2 = time()
        return tracker, t2-t1

    def eval_epoch(self, iter_name='val', final=False):
        """
        Evaluate the model on the test set.
        No backward computation is allowed.
        """
        t1 = time()
        self.model = self.model.eval()
        tracker = self.init_tracker()

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.iterators[iter_name]):
                tracker['total_bags'] += batch['rel'].size(0)
                tracker['total_sents'] += torch.sum(batch['bag_size'], dim=0).item()

                for keys in batch.keys():
                    if keys != 'bag_names':
                        batch[keys] = batch[keys].to(self.device)

                task_loss, rel_probs, kld, rec_loss, latent_z = self.model(batch)  # forward pass

                if self.config['reconstruction']:
                    loss = (self.config['task_weight'] * task_loss) + \
                           (1 - self.config['task_weight']) * (rec_loss['sum'] + kld)
                else:
                    loss = task_loss

                tracker['probabilities'] += [rel_probs.cpu().data.numpy()]
                tracker['gtruth'] += [batch['rel'].cpu().data.numpy()]
                tracker['total'] += [loss.item()]
                tracker['reco'] += [rec_loss['sum'].item()]
                tracker['ppl'] += [np.exp(rec_loss['mean'].item())]
                tracker['kld'] += [kld.item()]
                tracker['task'] += [task_loss.item()]
                tracker['kld_w'] += [1]

        t2 = time()
        return tracker, t2-t1

    def collect_codes(self, mode):
        """
        Collect the representations of the latent code.
        """
        self.model = self.model.eval()
        codes = {}
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.iterators[mode]):

                for keys in batch.keys():
                    if keys != 'bag_names':
                        batch[keys] = batch[keys].to(self.device)

                task_loss, rel_probs, kld, rec_loss, latent_z = self.model(batch)  # forward pass

                zs_split = torch.split(latent_z, batch['bag_size'].tolist(), dim=0)
                for z_bag, name in zip(zs_split, batch['bag_names']):
                    if name.split(' ### ')[2] != 'NA':
                        if name not in codes:
                            codes[name] = []
                        for i in range(0, z_bag.size(0)):
                            codes[name] += [z_bag[i].cpu().tolist()]

        with open(os.path.join(self.config['model_folder'], 'latent_mu_codes_'+mode+'.json'), 'w') as outfile:
            json.dump(codes, outfile)
