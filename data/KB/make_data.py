#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Adapted from 
https://github.com/billy-inn/HRERE/blob/master/data/prepare_data.py 
https://github.com/billy-inn/HRERE/blob/master/create_kg.py 
"""

import os
import requests
import tarfile
import numpy as np
import json
from tqdm import tqdm
import logging
import pandas as pd
import argparse


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


# Make data
def transform(x):
    if x == "/business/company/industry":
        return "/business/business_operation/industry"
    if x == "/business/company/locations":
        return "/organization/organization/locations"
    if x == "/business/company/founders":
        return "/organization/organization/founders"
    if x == "/business/company/major_shareholders":
        return "/organization/organization/founders"
    if x == "/business/company/advisors":
        return "/organization/organization/advisors"
    if x == "/business/company_shareholder/major_shareholder_of":
        return "/organization/organization_founder/organizations_founded"
    if x == "business/company/place_founded":
        return "/organization/organization/place_founded"
    if x == "/people/person/place_lived":
        return "/people/person/place_of_birth"
    if x == "/business/person/company":
        return "/organization/organization_founder/organizations_founded"
    return x


def load_data(args, kg_files):
    # Load train + validation
    df_train = []
    with open(args.train_file, 'r') as infile_train, open(args.val_file, 'r') as infile_val:
        for i, line in enumerate(tqdm(infile_train, desc='Loading training data')):
            a_dict = json.loads(line)
            if args.data == 'Freebase':
                df_train.append((a_dict['h']['id'], transform(a_dict['relation']), a_dict['t']['id']))
            else:
                df_train.append((a_dict['h']['id'], a_dict['relation'], a_dict['t']['id']))

        for j, line in enumerate(tqdm(infile_val, desc='Loading validation data')):
            a_dict = json.loads(line)
            if args.data == 'Freebase':
                df_train.append((a_dict['h']['id'], transform(a_dict['relation']), a_dict['t']['id']))
            else:
                df_train.append((a_dict['h']['id'], a_dict['relation'], a_dict['t']['id']))

    # Load test
    df_test = []
    with open(args.test_file, 'r') as infile:
        for k, line in enumerate(tqdm(infile, desc='Loading test data')):
            a_dict = json.loads(line)
            if args.data == 'Freebase':
                df_test.append((a_dict['h']['id'], transform(a_dict['relation']), a_dict['t']['id']))
            else:
                df_test.append((a_dict['h']['id'], a_dict['relation'], a_dict['t']['id']))

    # Read KB
    df_kg = []
    for kg_file in kg_files:
        with open(kg_file, 'r') as infile:
            for line in tqdm(infile, desc='Loading KG'):
                a1, r, a2 = line.rstrip().split('\t')
                df_kg.append((a1, r, a2))

    df_all = df_train + df_test
    return df_train, df_test, df_all, df_kg


def remove_test(df_test, df_kg):
    # Remove overlaps in pair link
    final_kg = []
    df_test_pairs = set([(a1, a2) for a1, r, a2 in df_test])
    logging.info('Test pairs (ignoring relation): {}'.format(len(df_test_pairs)))
    overlaps = 0
    for a1, r, a2 in tqdm(df_kg, desc='Discard test pairs from KG (independent of relation)', ascii=True):
        if (a1, a2) in df_test_pairs:
            overlaps += 1
            continue
        else:
            final_kg += [(a1, r, a2)]

    logging.info('Final KG pairs: {:.2f} % keep --> {} overlaps'.format(100 * len(final_kg) / len(df_kg), overlaps))
    return final_kg


def main(args):
    if not os.path.exists(args.data):
        os.makedirs(args.data)

    if args.data == 'Wikidata':
        logging.info("Downloading Wikidata dataset ...")
        os.system('wget https://www.dropbox.com/s/6sbhm0rwo4l73jq/wikidata5m_transductive.tar.gz?dl=1 -O wikidata5m_transductive.tar.gz')
        os.system('tar -xvf wikidata5m_transductive.tar.gz')
        os.system("mv wikidata5m_transductive_*.txt " + args.data)
        kg_files = [os.path.join(args.data, 'wikidata5m_transductive_train.txt'),
                    os.path.join(args.data, 'wikidata5m_transductive_valid.txt')]

    else:
        logging.info("Downloading fb3m dataset ...")
        download_file_from_google_drive("1XYdeovP9XuRh_j83R7sSow6mwQhoUdwQ", "fb3m.tar.gz")
        tar = tarfile.open("fb3m.tar.gz", "r:gz")
        tar.extractall()
        tar.close()
        os.remove("fb3m.tar.gz")
        os.system("mv fb3m/raw.txt " + os.path.join(args.data, "fb3m_triples.tsv"))
        os.system("rm -r fb3m/")
        kg_files = [os.path.join(args.data, 'fb3m_triples.tsv')]

    logging.info('Loading data ... ')
    df_train, df_test, df_all, df_kg = load_data(args, kg_files)
    final_kg = remove_test(df_test, df_kg)

    # --- Entities --- #
    test_entity_set = set([a1 for a1, r, a2 in df_test] + [a2 for a1, r, a2 in df_test])
    data_entity_set = set([a1 for a1, r, a2 in df_all] + [a2 for a1, r, a2 in df_all])
    train_entity_set = set([a1 for a1, r, a2 in df_train] + [a2 for a1, r, a2 in df_train])
    kg_entity_set = set([a1 for a1, r, a2 in final_kg] + [a2 for a1, r, a2 in final_kg])

    logging.info("%d entities in data" % len(data_entity_set))
    logging.info("%d entities in KG" % len(kg_entity_set))
    logging.info("%d entities in both" % len(data_entity_set.intersection(kg_entity_set)))
    logging.info("%d entities in TRAIN" % len(train_entity_set))
    logging.info("{:.2f} % entities in TRAIN+KG".format(
        100*len(train_entity_set.intersection(kg_entity_set)) / len(train_entity_set)))
    logging.info("{:.2f} % entities in TEST+KG".format(
        100 * len(test_entity_set.intersection(kg_entity_set)) / len(test_entity_set)))
    logging.info("=" * 50)

    # --- Relations --- #
    train_pairs = set([(a1, a2) for a1, r, a2 in df_train])
    pairs_intersection = sum([1 for (a1, a2) in train_pairs if ((a1 in kg_entity_set) and (a2 in kg_entity_set))])
    
    test_facts = set([(a1, r, a2) for a1, r, a2 in df_test if r != 'NA'])
    kg_facts = set(final_kg)
    logging.info("%d facts in test data" % len(test_facts))
    logging.info("%d facts in subgraph of KG" % len(kg_facts))
    logging.info("%d facts in both" % len(test_facts.intersection(kg_facts)))
    logging.info('{:.2f} % TRAIN pairs in KG (independent of relation)'.format(
        100 * pairs_intersection/len(train_pairs)))
    assert len(test_facts.intersection(kg_facts)) == 0, "Test facts exist in subgraph !!!"

    logging.info('Writing data to files ...')

    if args.data == 'Wikidata':
        with open(os.path.join(args.data, 'train.tsv'), 'w') as outfile:
            for arg1, rel, arg2 in final_kg:
                outfile.write('{}\t{}\t{}\n'.format(arg1, rel, arg2))

        with open(os.path.join(args.data, 'valid.tsv'), 'w') as outfile1, \
                open(os.path.join(args.data, 'test.tsv'), 'w') as outfile2:
            with open(os.path.join(args.data, 'wikidata5m_transductive_test.txt'), 'r') as infile:
                for line in infile:
                    arg1, rel, arg2 = line.rstrip().split('\t')
                    outfile1.write('{}\t{}\t{}\n'.format(arg1, rel, arg2))
                    outfile2.write('{}\t{}\t{}\n'.format(arg1, rel, arg2))

    else:
        with open(os.path.join(args.data, 'train.tsv'), 'w') as outfile:
            for arg1, rel, arg2 in final_kg[:-5000]:
                outfile.write('{}\t{}\t{}\n'.format(arg1, rel, arg2))

        with open(os.path.join(args.data, 'valid.tsv'), 'w') as outfile1, \
                open(os.path.join(args.data, 'test.tsv'), 'w') as outfile2:
            for arg1, rel, arg2 in final_kg[-5000:]:
                outfile1.write('{}\t{}\t{}\n'.format(arg1, rel, arg2))
                outfile2.write('{}\t{}\t{}\n'.format(arg1, rel, arg2))
            

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, choices=['Wikidata', 'Freebase'])
    parser.add_argument('--train_file', type=str)
    parser.add_argument('--val_file', type=str)
    parser.add_argument('--test_file', type=str)
    args = parser.parse_args()

    main(args)
