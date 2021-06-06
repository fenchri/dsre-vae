#!/usr/bin/python

import os, re, sys
import json
from tqdm import tqdm


def identify_argument(text, arg1):
    # find argument offsets
    if re.search(r'\b' + re.escape(arg1) + r'\b', text):
        start_offset_a = re.search(r'\b' + re.escape(arg1) + r'\b', text).start()

    elif re.search(re.escape(arg1) + r'\b', text):
        start_offset_a = re.search(re.escape(arg1) + r'\b', text).start()

    elif re.search(r'\b' + re.escape(arg1), text):
        start_offset_a = re.search(r'\b' + re.escape(arg1), text).start()

    else:
        assert False, 'Cannot find word == {}\n{}'.format(arg1, text)

    end_offset_a = start_offset_a + len(arg1)

    assert text[start_offset_a:end_offset_a] == arg1, \
        '{}\n{} <> Arg: {}'.format(text,
                              text[start_offset_a:end_offset_a], arg1)

    return (start_offset_a, end_offset_a)


with open(sys.argv[1], 'r') as infile, open(sys.argv[2], 'w') as outfile:
    for line in tqdm(infile):
        line = line.rstrip().split('\t')
        arg1_id = line[0]
        arg2_id = line[1]
        name1 = line[2]
        name2 = line[3]
        relation = line[4]
        sentence = line[5].replace(' ###END###', '')

        sentence = sentence.replace(name1, name1.replace('_', ' '))
        sentence = sentence.replace(name2, name2.replace('_', ' '))
        name1 = name1.replace('_', ' ')
        name2 = name2.replace('_', ' ')

        offsets1 = identify_argument(sentence, name1)
        offsets2 = identify_argument(sentence, name2)

        out_dict = {'text': sentence, 'relation': relation,
                    'h': {'name': name1, 'id': arg1_id, 'pos': list(offsets1)},
                    't': {'name': name2, 'id': arg2_id, 'pos': list(offsets2)}}

        outfile.write(json.dumps(out_dict) + '\n')
