#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 09/03/2020

author: fenia
"""

import os
import sys
import numpy as np
import yaml
from subprocess import check_output
import matplotlib as mpl
from sklearn.metrics import auc
mpl.use('Agg')
import matplotlib.pyplot as plt
import scipy.stats as stats


def load_config(file):
    with open(file, 'r') as stream:
        cfg = yaml.load(stream, Loader=yaml.FullLoader)
    return cfg


def wc(filename):
    return int(check_output(["wc", "-l", filename]).split()[0])


def humanized_time(second):
    """
    Args:
        second (float): time in seconds
    Returns: human readable time (hours, minutes, seconds)
    """
    m, s = divmod(second, 60)
    h, m = divmod(m, 60)
    return "%dh %02dm %02ds" % (h, m, s)


def print_performance(epoch, state, monitor, secs, name='train'):
    if name == 'train':
        print('---------- Epoch: {:02d} ----------'.format(epoch))

    template = '{:<5} | MAP = {:.04f} | LOSS = {:10.4f} | REC_LOSS = {:10.4f} | KLD = {:10.4f} | ' \
               'TASK_LOSS = {:10.4f} | KLD_weight = {:.04f} | PPL = {:10.4f} | {}'
    print(template.format(name.upper(),
                          monitor['pr_auc'],
                          state['total'] if state['total'] else 0.0,
                          state['reco'] if state['reco'] else 0.0,
                          state['kld'] if state['kld'] else 0.0,
                          state['task'],
                          state['kld_w'][-1], state['ppl'] if state['ppl'] else 0.0,
                          humanized_time(secs)))


def print_pr_curve(prec, rec, model_folder):
    base_list = ['BGWA', 'PCNN+ATT', 'PCNN', 'MIMLRE', 'MultiR', 'Mintz', 'RESIDE']
    color = ['tab:purple', 'tab:orange', 'tab:green', 'tab:blue', 'tab:pink', 'tab:cyan', 'tab:olive']
    marker = ['d', 's', '^', '*', 'v', 'x', '>']

    print()
    fig = plt.figure()
    for i, baseline in enumerate(base_list):
        precision = np.load('../pr_curves/' + baseline + '/precision.npy')
        recall = np.load('../pr_curves/' + baseline + '/recall.npy')
        print('{:<10} auc: {:.04f}'.format(baseline, auc(recall, precision)))
        plt.plot(recall, precision, color=color[i], label=baseline, lw=1, marker=marker[i], markevery=0.1, ms=6)

    print('---')
    print('{:<10} auc: {:.04f}\n'.format('Ours', auc(rec, prec)))
    plt.plot(rec[:], prec[:], color='tab:red', label='BiLSTM+SA', marker='o', markevery=0.1)
    plt.ylabel('Precision', fontsize=14)
    plt.xlabel('Recall', fontsize=14)
    plt.legend(loc="upper right", prop={'size': 10})
    plt.grid(True)
    plt.xlim([0.0, 0.45])
    plt.ylim([0.3, 1.0])
    fig.savefig(model_folder + '/pr_curve.png', bbox_inches='tight')


class Tee(object):
    """
    Object to print stdout to a file.
    """
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f_ in self.files:
            f_.write(obj)
            f_.flush()  # If you want the output to be visible immediately

    def flush(self):
        for f_ in self.files:
            f_.flush()


def str2bool(i):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(i, bool):
        return i
    if i.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif i.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def exp_name(params):
    exp = []
    exp.append('bs={}'.format(params['batch_size']))
    exp.append('bag={}'.format(params['bag_size']))
    exp.append('w={}'.format(params['word_embed_dim']))
    exp.append('r={}'.format(params['rel_embed_dim']))
    exp.append('z={}'.format(params['latent_dim']))
    exp.append('enc={}x{}'.format(params['enc_dim'], params['enc_layers']))
    exp.append('dec={}x{}'.format(params['dec_dim'], params['dec_layers']))
    exp.append('tf={}'.format(params['teacher_force']))
    exp.append('tw={}'.format(params['task_weight']))
    exp.append('lr={}'.format(params['lr']))
    exp.append('wd={}'.format(params['weight_decay']))
    exp.append('c={}'.format(params['clip']))
    exp.append('voc={}'.format(params['max_vocab_size']))
    if params['reconstruction']:
        exp.append('reco'+str(params['reconstruction']))
    if params['include_positions']:
        exp.append('pos')
    if params['priors']:
        exp.append('priors')
    if params['pretrained_embeds_file'] == '../embeds/glove.6B.50d.txt':
        exp.append('glove')
    exp = '_'.join(exp)
    return exp


def setup_log(params, folder_name=None, mode='train'):
    """
    Setup .log file to record training process and results.
    Args:
        params (dict): model parameters
    Returns:
        model_folder (str): model directory
    """
    if folder_name:
        model_folder = os.path.join(params['output_folder'], folder_name)
    else:
        model_folder = os.path.join(params['output_folder'], 'temp')

    experiment_name = exp_name(params)
    model_folder = os.path.join(model_folder, experiment_name)

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    log_file = os.path.join(model_folder, mode + '.log')

    f = open(log_file, 'w')
    sys.stdout = Tee(sys.stdout, f)
    return model_folder, experiment_name


def plot_latent(mu, variance, model_folder):
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)

    fig = plt.figure()
    for i in range(0, mu.size()):
        plt.plot(x, stats.norm.pdf(x, mu, sigma))
    fig.savefig(model_folder + '/latent_dist.png', bbox_inches='tight')


def print_options(params):
    print('''\nParameters:
            - Train/Val/Test    {}, {}, {}
            - Save folder       {}
            - batch_size        {}
            - Epoch             {}
            - Early stop        {}\tPatience = {}

            - Word Embeds       {:<5}\tDim: {}\tFreeze: {}
            - Rel embeds dim    {}
            - Pos embed dim     {}\tUse: {}
            - Latent dim        {}\tReconstruction: {}\tPriors: {}
            - Encoder Dim       {}\tLayers {}
            - Decoder Dim       {}\tLayers {}
            - Weight Decay      {}
            - Gradient Clip     {}
            - Dropout I         {}
            - Dropout O         {}
            - Learning rate     {}
            - Bag size          {} (0 means take all)
            - Vocab size        {}
            - Max sent len      {}
            - Teacher force     {}
            - Task Loss weight  {}
            '''.format(params['train_data'], params['val_data'], params['test_data'], params['output_folder'],
                       params['batch_size'], params['epochs'], params['early_stop'], params['patience'],
                       params['pretrained_embeds_file'] if params['pretrained_embeds_file'] else 'None',
                       params['word_embed_dim'], params['freeze_words'],
                       params['rel_embed_dim'], params['pos_embed_dim'], params['include_positions'],
                       params['latent_dim'], params['reconstruction'], params['priors'],
                       params['enc_dim'], params['enc_layers'], params['dec_dim'], params['dec_layers'],
                       params['weight_decay'], params['clip'],
                       params['input_dropout'], params['output_dropout'], params['lr'],
                       params['bag_size'], params['max_vocab_size'], params['max_sent_len'],
                       params['teacher_force'], params['task_weight']))


def histogram(x):
    fig = plt.figure()
    plt.hist(x, np.arange(x.shape[0]))
    fig.savefig('word_freq_hist.png', bbox_inches='tight')
