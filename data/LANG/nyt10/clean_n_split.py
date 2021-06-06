#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 11/09/2020

author: fenia
"""

import sys
import json
import numpy as np
from sklearn.model_selection import train_test_split
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--input_data', type=str)
parser.add_argument('--train_data', type=str)
parser.add_argument('--r2id', type=str)
parser.add_argument('--val_data', type=str)
args = parser.parse_args()


with open(args.r2id, 'r') as infile:
    relations = json.load(infile)
    relations = list(relations.keys())

train_bags = {}
instances = 0
with open(args.input_data, 'r') as infile:
    for line in infile:
        line = json.loads(line)
        instances += 1

        tmp = [line['text'], line['h']['pos'], line['t']['pos'], line['h']['name'], line['t']['name']]
        triple = (line['h']['id'], line['relation'], line['t']['id'])

        if line['relation'] not in relations:
            triple = (line['h']['id'], 'NA', line['t']['id'])

        if triple not in train_bags:
            train_bags[triple] = [tmp]
        else:
            train_bags[triple] += [tmp]

print('Train instances:  {}'.format(instances))
print('TRAIN bags:       {}'.format(len(train_bags)))
print('Unique labels:    {}'.format(len(list(set(relations)))))

counts = {}
for r in relations:
    counts[r] = 0
for b in train_bags:
    counts[b[1]] += 1

exclude_items = {}
keep_items = {}
new_relations = []
for b in train_bags:
    if counts[b[1]] == 1:
        exclude_items[b] = train_bags[b]
    else:
        keep_items[b] = train_bags[b]
        new_relations += [b[1]]

print('TRAIN (relations > 2): {}'.format(len(keep_items)))
print('TRAIN labels (relations > 2): {}'.format(len(new_relations)))

print('Splitting training set into train and validation (90/10) ...')
X_train, X_val, y_train, y_val = train_test_split(np.arange(len(keep_items)),
                                                  new_relations,
                                                  test_size=0.10,
                                                  random_state=42,
                                                  stratify=new_relations)

print("Storing files ... ")

with open(args.train_data, 'w') as outfile:
    for i, ins in enumerate(keep_items):
        if i in X_train:
            for tmp in keep_items[ins]:
                line = {'text': tmp[0], 'h': {'pos': tmp[1], 'id': ins[0], 'name': tmp[3]},
                        't': {'pos': tmp[2], 'id': ins[2], 'name': tmp[4]}, 'relation': ins[1]}
                outfile.write(json.dumps(line))
                outfile.write('\n')

    # Add unis
    for ins in exclude_items:
        for tmp in exclude_items[ins]:
            line = {'text': tmp[0], 'h': {'pos': tmp[1], 'id': ins[0], 'name': tmp[3]},
                    't': {'pos': tmp[2], 'id': ins[2], 'name': tmp[4]}, 'relation': ins[1]}
            outfile.write(json.dumps(line))
            outfile.write('\n')

with open(args.val_data, 'w') as outfile:
    for i, ins in enumerate(keep_items):
        if i in X_val:
            for tmp in keep_items[ins]:
                line = {'text': tmp[0], 'h': {'pos': tmp[1], 'id': ins[0], 'name': tmp[3]},
                        't': {'pos': tmp[2], 'id': ins[2], 'name': tmp[4]}, 'relation': ins[1]}
                outfile.write(json.dumps(line))
                outfile.write('\n')
