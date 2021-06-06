#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 02/09/2020

author: fenia
"""

import re
import numpy as np
from tqdm import tqdm
import os
import argparse
import json
import random

np.random.seed(2021)              # Numpy module
random.seed(2021)


class Preprocessor:
    """ Class for pre-processing the data """
    def __init__(self, max_sent_len, lowercase=True):
        self.max_sent_len = max_sent_len
        self.lowercase = lowercase
        self.track_instances = {}
        self.duplicates = 0
        self.outliers = 0

    @staticmethod
    def zero_digits(s):
        """
        Replace every digit in a string by a hash
        """
        return re.sub('\d', '#', s)

    def calc_position(self, tokens, offsets_head, offsets_tail):
        head_start, head_end = offsets_head
        tail_start, tail_end = offsets_tail

        e1 = self.e2tok(tokens, head_start, head_end)
        e2 = self.e2tok(tokens, tail_start, tail_end)

        assert e1 and e2, '{}\t{}-{}\t{}\t{}-{}\t{}'.format(tokens, head_start, head_end, e1,
                                                            tail_start, tail_end, e2)
        return e1, e2

    @staticmethod
    def using_split2(line, _len=len):
        """
        Credits to https://stackoverflow.com/users/1235039/aquavitae
        :param line: sentence
        :return: a list of words and their indexes in a string.
        """
        words = line.split(' ')
        index = line.index
        offsets = []
        append = offsets.append
        running_offset = 0
        for word in words:
            word_offset = index(word, running_offset)
            word_len = _len(word)
            running_offset = word_offset + word_len
            append((word, word_offset, running_offset))
        return offsets

    def e2tok(self, text, estart, eend):
        """
        convert entity offsets to token ids
        """
        span2append = []
        text_ = self.using_split2(text)
        for tok_id, (tok, start, end) in enumerate(text_):
            start = int(start)
            end = int(end)

            if (start, end) == (estart, eend):
                span2append.append(tok_id)
            elif start == estart and end < eend:
                span2append.append(tok_id)
            elif start > estart and end < eend:
                span2append.append(tok_id)
            elif start > estart and end == eend:
                span2append.append(tok_id)

            # entity has more characters (incomplete tokenization)
            elif len(set(range(start, end)).intersection(set(range(estart, eend)))) > 0:
                span2append.append(tok_id)

        # include all tokens!
        if len(span2append) == len(text[estart:eend].split(' ')):
            return span2append
        else:
            new_span2append = []
            for sp in span2append:
                if text.split(' ')[sp] == '':
                    continue
                else:
                    new_span2append += [sp]

            if len(new_span2append) == len(text[estart:eend].split(' ')):
                return new_span2append
            else:
                return new_span2append

    def resize_sentence(self, sentence, id_a, id_b):
        x = min(id_a + id_b)
        y = max(id_a + id_b)

        if len(sentence[x:y + 1]) < self.max_sent_len:
            remaining_words = self.max_sent_len - len(sentence[x:y + 1])
            left = remaining_words // 2
            right = remaining_words // 2
            if left > 5:
                left = 5
            if right > 5:
                right = 5
        else:
            left, right = 0, 0

        sentence_new = sentence[x:y + 1]

        if len(sentence[:x]) > left > 0:
            start = left
            sentence_new = sentence[x-left:x] + sentence_new
        elif len(sentence[:x]) <= left > 0:
            start = len(sentence[:x])
            sentence_new = sentence[:x] + sentence_new
        else:
            start = 0

        if len(sentence[y+1:]) > right > 0:
            sentence_new = sentence_new + sentence[y+1:y+1+right]
        elif len(sentence[y+1:]) <= right > 0:
            sentence_new = sentence_new + sentence[y+1:]

        id_a_new = [i - x + start for i in id_a]
        id_b_new = [j - x + start for j in id_b]

        return sentence_new, id_a_new, id_b_new

    def check_outlier(self, sentence, ids1, ids2):
        if (ids1[-1] >= self.max_sent_len) or (ids2[-1] >= self.max_sent_len):
            sentence = sentence.split(' ')
            arg1 = sentence[ids1[0]:ids1[-1]+1]
            arg2 = sentence[ids2[0]:ids2[-1]+1]

            sentence_new, ids1_new, ids2_new = self.resize_sentence(sentence, ids1, ids2)

            # print(sentence_new, ids1_new, ids2_new, ids1, ids2)
            assert sentence_new[ids1_new[0]:ids1_new[-1]+1] == arg1, '{} <> {}\n{}\n{}-{}-{}'.format(
                sentence_new[ids1_new[0]:ids1_new[-1] + 1], arg1, sentence_new, arg2, ids1_new, ids2_new)
            assert sentence_new[ids2_new[0]:ids2_new[-1]+1] == arg2, '{} <> {}\n{}\n{}-{}-{}'.format(
                sentence_new[ids2_new[0]:ids2_new[-1] + 1], arg2, sentence_new, arg1, ids1_new, ids2_new)

            # if it is bad again ...
            if (ids1_new[-1] >= self.max_sent_len) or (ids2_new[-1] >= self.max_sent_len):
                return 'reject'
            else:
                if ids1_new[0] == 0:
                    a = 0
                else:
                    a = len(' '.join(sentence_new[0:ids1_new[0]])) + 1
                if ids2_new[0] == 0:
                    b = 0
                else:
                    b = len(' '.join(sentence_new[0:ids2_new[0]])) + 1
                off1 = [a, a+len(' '.join(arg1))]
                off2 = [b, b+len(' '.join(arg2))]
                sentence_new = ' '.join(sentence_new)
                assert sentence_new[off1[0]:off1[1]] == ' '.join(arg1), '{} <> {}'.format(
                    repr(sentence_new[off1[0]:off1[1]]), arg1)
                assert sentence_new[off2[0]:off2[1]] == ' '.join(arg2)
                return [sentence_new, off1, off2]
        else:
            return 'accept'

    def remove_duplicates(self, item):
        sth = tuple((item['text'], item['h']['name'], item['h']['id'], item['t']['name'], item['t']['id']))

        if sth in self.track_instances:
            self.track_instances[sth] += 1
            return True
        else:
            self.track_instances[sth] = 1
            return False

    def process(self, item, mode='train'):
        sentence = item['text']
        arg1, arg2 = item['h']['name'], item['t']['name']
        offsets_head, offsets_tail = item['h']['pos'], item['t']['pos']

        arg1_toks, arg2_toks = self.calc_position(sentence, offsets_head, offsets_tail)

        if self.lowercase:
            sentence = sentence.lower()
            arg1 = arg1.lower()
            arg2 = arg2.lower()
        sentence = self.zero_digits(sentence)
        arg1 = self.zero_digits(arg1)
        arg2 = self.zero_digits(arg2)

        if mode == 'train':
            check = self.check_outlier(sentence, arg1_toks, arg2_toks)

            if check == 'accept':
                return {'text': sentence, 'relation': item['relation'],
                        'h': {'name': arg1, 'id': item['h']['id'], 'toks': arg1_toks},
                        't': {'name': arg2, 'id': item['t']['id'], 'toks': arg2_toks}}

            elif check == 'reject':
                self.outliers += 1
                return None

            else:
                # return reduced list
                arg1_toks, arg2_toks = self.calc_position(check[0], check[1], check[2])
                return {'text': check[0], 'relation': item['relation'],
                        'h': {'name': arg1, 'id': item['h']['id'], 'toks': arg1_toks},
                        't': {'name': arg2, 'id': item['t']['id'], 'toks': arg2_toks}}

        else:
            if len(sentence) > 9000:
                print('Sentence too long (> 9000)! Ignoring ...')
                return None
            else:
                return {'text': sentence, 'relation': item['relation'],
                        'h': {'name': arg1, 'id': item['h']['id'], 'toks': arg1_toks},
                        't': {'name': arg2, 'id': item['t']['id'], 'toks': arg2_toks}}


def main(args, file_path, mode):
    preprocessor = Preprocessor(args.max_sent_len, args.lowercase)

    bag_scope, bag_labels, bag_offsets = {}, {}, {}
    instances = 0
    original_instances = 0
    bag_reduction = 0
    facts = {}
    negatives = {}
    with open(file_path) as infile:
        for item in tqdm(infile, desc='Processing ' + mode.upper()):
            item = json.loads(item)

            original_instances += 1
            sample = preprocessor.process(item, mode)

            if sample:
                name = sample['h']['id'] + ' ### ' + sample['t']['id']
                fact = (sample['h']['id'], sample['t']['id'], sample['relation'])

                if item['relation'] != 'NA' and fact not in facts:
                    facts[fact] = 1
                elif item['relation'] != 'NA':
                    facts[fact] += 1
                elif item['relation'] == 'NA' and fact not in negatives:
                    negatives[fact] = 1
                elif item['relation'] == 'NA':
                    negatives[fact] += 1

                if name not in bag_scope:  # new bag
                    bag_scope[name] = []
                    bag_labels[name] = []
                    bag_offsets[name] = []

                bag_labels[name].append(sample['relation'])
                if mode == 'train':
                    if sample['text'] in bag_scope[name]:
                        preprocessor.duplicates += 1
                    else:
                        instances += 1
                        bag_scope[name].append(sample['text'])
                        bag_offsets[name].append({
                            'h': {'name': sample['h']['name'],
                                  'tokens': sample['h']['toks']},
                            't': {'name': sample['t']['name'],
                                  'tokens': sample['t']['toks']}
                        })
                else:
                    instances += 1
                    bag_scope[name].append(sample['text'])
                    bag_offsets[name].append({
                        'h': {'name': sample['h']['name'],
                              'tokens': sample['h']['toks']},
                        't': {'name': sample['t']['name'],
                              'tokens': sample['t']['toks']}
                    })

    # reduce bag_size
    if mode == 'train':
        for bn in bag_scope.keys():
            bag_s = bag_scope[bn]
            bag_o = bag_offsets[bn]
            if 0 < args.max_bag_size < len(bag_s):
                choices = np.array(random.sample(range(len(bag_s)), args.max_bag_size))  # randomly select instances
                bag_scope[bn] = [b for i, b in enumerate(bag_s) if i in choices]
                bag_offsets[bn] = [b for i, b in enumerate(bag_o) if i in choices]
                bag_reduction += args.max_bag_size
            else:
                bag_reduction += len(bag_s)

    # write
    print('Original instances:  ', original_instances)
    print('Final instances:     ', instances)
    print('Total Bags:          ', len(bag_scope))
    print('Duplicates:          ', preprocessor.duplicates)
    print('Outliers:            ', preprocessor.outliers)
    print('After bag reduction: ', bag_reduction)

    with open(os.path.join(args.path, 'processed', args.dataset + '_' + mode + '.txt'), 'w') as outfile:
        for bn in tqdm(bag_scope.keys(), desc='Saving'):
            labels = list(set(bag_labels[bn]))
            if len(labels) > 1 and 'NA' in labels:  # pairs with NA + some relation
                labels = [r for r in labels if r != 'NA']

            sents = []
            for txt, rest in zip(bag_scope[bn], bag_offsets[bn]):
                temp = {'text': txt}
                temp.update(rest)
                sents += [temp]

            write_dict = {'bag_name': bn, 'sentences': sents, 'bag_labels': labels}
            outfile.write(json.dumps(write_dict) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_sent_len', type=int)
    parser.add_argument('--lowercase', action="store_true")
    parser.add_argument('--max_bag_size', type=int, default=500)
    parser.add_argument('--path', type=str)
    parser.add_argument('--dataset', type=str, choices=['nyt10', 'nyt10_570k', 'wiki_distant'])
    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.path, 'processed')):
        os.makedirs(os.path.join(args.path, 'processed'))

    for key in ['train', 'val', 'test']:
        if args.dataset == 'nyt10_570k' and key == 'test':
            filepath = os.path.join('nyt10', 'nyt10' + '_' + key + '.txt')
        else:
            filepath = os.path.join(args.path, args.dataset + '_' + key + '.txt')
        main(args, filepath, key)
