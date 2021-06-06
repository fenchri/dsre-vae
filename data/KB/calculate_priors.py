#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14-Sep-2020
author: fenia
"""

import os
import json
import numpy as np
import argparse
from tqdm import tqdm
import pickle


def main(args):

    print('Loading KG embeddings ...')

    entity_embeddings = np.load(args.kg_embeds)
    print(entity_embeddings)

    id2entity, entity2id = {}, {}
    with open(args.e_map, 'r') as infile:
        for line in infile:
            line = line.rstrip().split('\t')
            id2entity[int(line[0])] = line[1]
            entity2id[line[1]] = int(line[0])

    print('Number of entity embeddings: {}'.format(entity_embeddings.shape))

    mus = {}
    dim = entity_embeddings.shape[1]
    total_pairs = 0
    for filef in args.data:
        with open(filef, 'r') as infile:
            for line in infile:
                line = json.loads(line)
                
                arg1 = line['h']['id']
                arg2 = line['t']['id']
                total_pairs += 1
                if (arg1 in entity2id) and (arg2 in entity2id):
                    arg1_embed = entity_embeddings[entity2id[arg1]]
                    arg2_embed = entity_embeddings[entity2id[arg2]]
                    
                    mus[arg1 + ' ### ' + arg2] = (arg1_embed - arg2_embed).tolist()

    print('Total priors: {} / Total pairs: {}'.format(len(mus), total_pairs))
    with open(os.path.join(args.kg, 'pairs_mu_'+str(dim)+'d.json'), 'w', encoding='utf-8') as outfile:
        json.dump(mus, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--kg_embeds', type=str)
    parser.add_argument('--e_map', type=str)
    parser.add_argument('--data', type=str, nargs='*')
    parser.add_argument('--kg', type=str, choices=['Freebase', 'Wikidata', 'Freebase_570k'])
    args = parser.parse_args()

    if not os.path.exists(args.kg):
        os.makedirs(args.kg)
    main(args)
