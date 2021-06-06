#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 08/06/2020

author: fenia
"""

import torch
from torch import nn, torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


class SelectiveAttention(nn.Module):
    """
    Simply bag_sents * relations (no learned parameters)
    """
    def __init__(self, device):
        super(SelectiveAttention, self).__init__()

        self.softmax = nn.Softmax(dim=1)  # sentence dimension
        self.device = device

    def masking(self, bag, bag_size):
        # mask padding elements
        tmp = torch.arange(bag.size(1)).repeat(bag.size(0), 1).unsqueeze(-1).to(self.device)
        mask = torch.lt(tmp, bag_size[:, None, None].repeat(1, tmp.size(1), 1))
        return mask

    def forward(self, bags, bags_size, relation_embeds):
        mask = self.masking(bags, bags_size)
        scores = torch.matmul(bags, relation_embeds.transpose(0, 1))
        scores = torch.where(mask, scores, torch.full_like(scores, float('-inf')).to(self.device))
        scores = self.softmax(scores)

        sent_rep = torch.matmul(scores.permute(0, 2, 1), bags)  # for each bag: 1 representation for each relation
        return sent_rep
