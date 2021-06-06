#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 31/08/2020

author: fenia
"""

import torch
from torch.utils.data import Dataset, Sampler
import heapq
import math
import numpy as np


class BatchBagSampler(Sampler):
    """
    Defines a strategy for drawing batches of samples from the dataset.
    """

    def __init__(self, bag_sizes, batch_size, max_bag_size, shuffle=False, drop_last=False):
        self.shuffle = shuffle
        self.drop_last = drop_last

        # bag_sizes = np.array([len(b) for b in bag_sizes])
        bag_sizes = np.array([len(b['text']) for name, b in bag_sizes.items()])
        if max_bag_size != 0:
            bag_sizes[bag_sizes > max_bag_size] = max_bag_size

        num_sections = math.ceil(len(bag_sizes) / batch_size)

        # create batches with approximately the same number of total sentences
        # Credits: https://stackoverflow.com/questions/61648065/split-list-into-n-sublists-with-approximately-equal-sums
        self.batches = [[] for _ in range(num_sections)]
        totals = [(0, i) for i in range(num_sections)]
        heapq.heapify(totals)
        for i, value in enumerate(bag_sizes):
            total, index = heapq.heappop(totals)
            self.batches[index].append(i)
            heapq.heappush(totals, (total + value, index))

    def __iter__(self):
        if self.shuffle:
            return iter(self.batches[i] for i in torch.randperm(len(self.batches)))
        else:
            return iter(self.batches)

    def __len__(self):
        return len(self.batches)
