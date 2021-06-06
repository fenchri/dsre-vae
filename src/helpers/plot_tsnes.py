#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 13/09/2020

author: fenia
"""

import os
from sklearn.manifold import TSNE
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
from tqdm import tqdm
from collections import Counter
import pandas as pd
import random
import argparse
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import json
random.seed(42)
np.random.seed(42)


def load_priors(filef):
    print('Loading mu vectors for pairs: {}'.format(filef))
    with open(filef, 'r') as infile:
        priors = json.load(infile)
    return priors


def plot_tsne_2d(data_vecs, data_labels, target, args, mode='priors'):
    """
    Plot t-SNE 2D
    """
    if args.dataset == 'nyt10':
        target = ['nationality', 'contains', 'place_of_birth', 'place_lived', 'place_of_death',
                  'neighborhood_of', 'company', 'admin_divisions', 'country', 'capital']

    tsne_em = TSNE(n_components=2, verbose=1, random_state=42).fit_transform(data_vecs)
    sns.set_style("whitegrid")
    sns.set_context("paper")
    fig = plt.figure(figsize=(8, 6))

    print('Plotting T-SNE ...')
    total_data = pd.DataFrame(list(zip(tsne_em[:, 0], tsne_em[:, 1], data_labels)),
                              columns=['dim1', 'dim2', 'label'])

    ax = sns.scatterplot(x="dim1", y="dim2", data=total_data, hue='label', hue_order=target, linewidth=0,
                         palette=sns.color_palette('muted'))

    ax.grid(b=True, which='major', color='lightgrey', linewidth=0.5)

    plt.xlabel("")
    plt.ylabel("")
    if args.legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[0:], labels=labels[0:])
        plt.setp(ax.get_legend().get_texts(), fontsize='8')
        plt.legend(loc='lower right')
    else:
        ax.get_legend().remove()

    fig.savefig(f'../../plots/tsne_{args.dataset}_{args.dim}d_{mode}_{args.split}_2D.png', bbox_inches='tight')
    fig.savefig(f'../../plots/tsne_{args.dataset}_{args.dim}d_{mode}_{args.split}_2D.pdf', bbox_inches='tight')
    print(f'Figure saved at ../../plots/tsne_{args.dataset}_{args.dim}d_{mode}_2D.png')


def fix_labels(data_labels, target_relations):
    new_data_labels = []
    for l in data_labels:
        new_l = l.split('/')[-1]
        if new_l == 'administrative_divisions':
            new_data_labels += ['admin_divisions']
        else:
            new_data_labels += [new_l]

    new_target_relations = []
    for l in target_relations:
        new_l = l.split('/')[-1]
        if new_l == 'administrative_divisions':
            new_target_relations += ['admin_divisions']
        else:
            new_target_relations += [new_l]

    return new_data_labels, new_target_relations


def load_data(args):
    facts = {}
    dist = {}
    with open(args.filename, 'r') as infile:
        for item in tqdm(infile, desc='Loading dataset file'):
            sample = json.loads(item)

            if sample['h']['id'] + ' ### ' + sample['t']['id'] not in facts:
                facts[sample['h']['id'] + ' ### ' + sample['t']['id']] = [sample['relation']]

            else:
                if sample['relation'] not in facts[sample['h']['id'] + ' ### ' + sample['t']['id']]:
                    facts[sample['h']['id'] + ' ### ' + sample['t']['id']] += [sample['relation']]

            if sample['relation'] not in dist:
                dist[sample['relation']] = 1
            else:
                dist[sample['relation']] += 1

    dist = {k: v for k, v in sorted(dist.items(), key=lambda item: item[1], reverse=True)}
    target_relations = [k for i, (k, v) in enumerate(dist.items()) if i < 11 and k != 'NA']
    print(target_relations)
    assert len(target_relations) == 10
    return facts, dist, target_relations


def plot_posteriors(args):
    """
    Plot Posteriors (After training VAE with priors)
    """
    facts, dist, target_relations = load_data(args)

    with open(args.posteriors, 'r') as infile:
        posteriors = json.load(infile)

    vectors = []
    labels = []
    rel_counts = {}
    for t in target_relations:
        rel_counts[t] = 0

    for name in posteriors.keys():
        post = posteriors[name]
        arg1, arg2, rel = name.split(' ### ')
        choose = 2
        if rel in target_relations:
            if len(post) > choose:
                random_indices = np.random.choice(len(post), size=choose, replace=False)
            else:
                random_indices = list(range(len(post)))

            if rel == '/location/location/contains':
                if rel_counts[rel] < 3000:
                    sents = np.array(post)[random_indices, :]
                    vectors += [sents]
                    if len(post) < choose:
                        labels.extend([rel] * len(post))
                        rel_counts[rel] += len(post)
                    else:
                        labels.extend([rel] * choose)
                        rel_counts[rel] += choose

            elif rel_counts[rel] < 3000:
                sents = np.array(post)[random_indices, :]
                vectors += [sents]
                if len(posteriors[name]) < choose:
                    labels.extend([rel] * len(post))
                    rel_counts[rel] += len(post)
                else:
                    labels.extend([rel] * choose)
                    rel_counts[rel] += choose

    vectors = np.vstack(vectors)
    print(vectors.shape)
    new_data_labels, new_target_relations = fix_labels(labels, target_relations)
    print(len(new_data_labels))

    return vectors, new_data_labels, new_target_relations


def plot_priors(args):
    """
    Plot Priors (after TransE)
    """
    prior_data = load_priors(args.priors)
    facts, dist, target_relations = load_data(args)

    print('Collecting vectors ...')
    vectors = []
    data_labels = []
    choose = {}
    for t in target_relations:
        choose[t] = 0

    for f in facts.keys():
        final_fact = facts[f][0]
        if (f in prior_data) and (final_fact in target_relations):
            if final_fact == '/location/location/contains':
                if choose[final_fact] < 2000:
                    choose[final_fact] += 1
                    vectors += [np.array(prior_data[f])]
                    data_labels += [final_fact]

            else:
                if choose[final_fact] < 2000:
                    choose[final_fact] += 1
                    vectors += [np.array(prior_data[f])]
                    data_labels += [final_fact]

    vectors = np.vstack(vectors)
    new_data_labels, new_target_relations = fix_labels(data_labels, target_relations)
    assert len(new_target_relations) == 10

    return vectors, new_data_labels, new_target_relations


def main(args):
    if args.priors:
        vectors, new_data_labels, new_target_relations = plot_priors(args)
        print('Total vectors plot plot: {}'.format(vectors.shape[0]))
        print(new_target_relations)
        plot_tsne_2d(vectors, new_data_labels, new_target_relations, args, mode='priors')

    if args.posteriors:
        vectors, new_data_labels, new_target_relations = plot_posteriors(args)
        print('Total vectors plot plot: {}'.format(vectors.shape[0]))
        print(new_target_relations)
        plot_tsne_2d(vectors, new_data_labels, new_target_relations, args, mode='posteriors')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--priors', type=str)
    parser.add_argument('--posteriors', type=str)
    parser.add_argument('--dataset', choices=['nyt10', 'wikidistant'])
    parser.add_argument('--filename', type=str, default='../../data/LANG/wikidistant/wiki_distant_train.txt')
    parser.add_argument('--legend', action='store_true')
    parser.add_argument('--split', type=str)
    args = parser.parse_args()

    if not os.path.exists('../../plots/'):
        os.makedirs('../../plots/')

    main(args)
