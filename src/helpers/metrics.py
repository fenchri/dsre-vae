#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 30-April-2020
author: fenia
"""

import sys
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_recall_fscore_support, auc
from sklearn.preprocessing import label_binarize
import numpy as np

np.set_printoptions(threshold=sys.maxsize)


def accuracy(truth, prediction):
    pos_total = np.sum(truth != 0)
    pos_correct = np.sum(np.array(prediction == truth, 'f') * np.array(truth != 0, 'f'))
    pos_acc = float(pos_correct) / float(pos_total)
    return pos_acc


def get1hot_gt(ground_truth, num_class):
    one_hot = np.zeros((len(ground_truth), num_class), 'i')
    for i, gt in enumerate(ground_truth):
        one_hot[i, gt] = 1
    return one_hot


def get1hot_pred(preds, num_class):
    one_hot = np.zeros((len(preds), num_class), 'i')
    for i, p in enumerate(preds):
        for j, p_ in enumerate(p):
            if p_ >= 0.5:
                one_hot[i, j] = 1
    return one_hot


def fbeta_score(p, r, beta=1.0):
    beta_square = beta * beta
    if (p != 0.0) and (r != 0.0):
        res = ((1 + beta_square) * p * r / (beta_square * p + r))
    else:
        res = 0.0
    return res


def prf1_micro(y_true, y_scores):
    prediction = np.where(y_scores >= 0.5, 1, 0)  # map to one-hot vector

    pr, rec, f1, sup = precision_recall_fscore_support(y_true[:, 1:], prediction[:, 1:], average='micro')
    return pr, rec, f1


def map_score(y_true, y_scores):
    """
    Mean Average Precision Score: AUC for PR curve
    """
    return average_precision_score(y_true, y_scores)


def p_at_n(y_true, y_scores):
    """
    Precision @ N
    Sort predictions according to probability.
    Estimate precision for the top N (this is micro-averaged)
    """
    def n_score(n):
        corr_num = 0.0
        for i in order[:n]:
            corr_num += 1.0 if (y_true[i] == 1) else 0
        return corr_num / n

    order = np.argsort(-y_scores)
    return n_score(100), n_score(200), n_score(300), n_score(500)


def pr_rec_curve(y_true, y_scores):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    return precision, recall


def calc_all_perf(y_true, y_scores, nclasses, mode):
    # Micro P/R/F1
    # p, r, f1 = prf1_micro(y_true, y_scores)
    acc = accuracy(y_true, y_scores)

    # Binarize for P@N and PR-AUC if not already
    y_true = np.reshape(y_true[:, 1:], (-1))  # ignore NA category predictions
    y_scores = np.reshape(y_scores[:, 1:], (-1))

    if mode != 'train':
        patn = p_at_n(y_true, y_scores)
        pr_auc = map_score(y_true, y_scores)
        p_points, r_points = pr_rec_curve(y_true, y_scores)
    else:
        patn, pr_auc = [0, 0, 0, 0], 0
        p_points, r_points = [], []
    perf = {'pr_auc': pr_auc,
            'p@100': patn[0], 'p@200': patn[1], 'p@300': patn[2], 'p@500': patn[3]}
    return perf, p_points, r_points
