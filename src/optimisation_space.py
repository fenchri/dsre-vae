#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15/06/2020

author: fenia
"""


train_data_wiki = '../data/LANG/wiki_distant/processed/wiki_distant_train.txt'
val_data_wiki = '../data/LANG/wiki_distant/processed/wiki_distant_val.txt'
test_data_wiki = '../data/LANG/wiki_distant/processed/wiki_distant_test.txt'
rels_wiki = '../data/LANG/wiki_distant/wiki_distant_rel2id.json'

train_data_nyt = '../data/LANG/nyt10/processed/nyt10_train.txt'
val_data_nyt = '../data/LANG/nyt10/processed/nyt10_val.txt'
test_data_nyt = '../data/LANG/nyt10/processed/nyt10_test.txt'
rels_nyt = '../data/LANG/nyt10/nyt10_rel2id.json'


def model_space_reco(trial, model_name):
    space_reco = {'train_data': train_data_nyt if model_name == 'nyt' else train_data_wiki,
                  'val_data': val_data_nyt if model_name == 'nyt' else val_data_wiki,
                  'test_data': test_data_nyt if model_name == 'nyt' else test_data_wiki,
                  'relations_file': rels_nyt if model_name == 'nyt' else rels_wiki,
                  'output_folder': '../saved_models_tuning/',
                  'pretrained_embeds_file': '../embeds/glove.6B.50d.txt',
                  'model_folder': '../saved_models_tuning/nyt10_reco/',
                  'enc_dim': 256,
                  'dec_dim': 256,
                  'word_embed_dim': 50,
                  'rel_embed_dim': 64,
                  'pos_embed_dim': 8,
                  'latent_dim': 64,
                  'input_dropout': trial.suggest_discrete_uniform('input_dropout', 0.0, 1.0, 0.05),
                  'output_dropout': trial.suggest_discrete_uniform('output_dropout', 0.0, 1.0, 0.05),
                  'weight_decay': trial.suggest_categorical('weight_decay', [0.0, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]),
                  'lr': 0.001,
                  'clip': trial.suggest_categorical('clip', [0, 1, 5, 10]),
                  'batch_size': 128,
                  'max_vocab_size': 40000 if model_name == 'nyt' else 50000,
                  'epochs': 100,
                  'device': 0,
                  'early_stop': True,
                  'patience': 5,
                  'runs': 1,
                  'max_sent_len': 50,
                  'show_example': False,
                  'primary_metric': 'pr_auc',
                  'bag_size': 500,
                  'teacher_force': trial.suggest_discrete_uniform('teacher_force', 0.2, 0.5, 0.05),
                  'enc_layers': 1,
                  'dec_layers': 1,
                  'enc_bidirectional': True,
                  'dec_bidirectional': False,
                  'optimizer': 'Adam',
                  'include_positions': True,
                  'lowercase': True,
                  'priors': False,
                  'task_weight': trial.suggest_discrete_uniform('task_weight', 0.5, 0.9, 0.05),
                  'freeze_words': False,
                  'reconstruction': True,
                  'log_interval': 100,
                  'reco_loss': 'AdaptiveSoftmax'}
    return space_reco


def model_space_prior(trial):
    space_prior = {'train_data': train_data_nyt if model_name == 'nyt' else train_data_wiki,
                   'val_data': val_data_nyt if model_name == 'nyt' else val_data_wiki,
                   'test_data': test_data_nyt if model_name == 'nyt' else test_data_wiki,
                   'relations_file': rels_nyt if model_name == 'nyt' else rels_wiki,
                   'output_folder': '../saved_modes_tuning/',
                   'pretrained_embeds_file': '../embeds/glove.6B.50d.txt',
                   'prior_mus_file': '../data/KB/Freebase/pairs_mu_64d.json' if model_name == 'nyt' else '../data/KB/Wikidata/pairs_mu_64d.json',
                   'enc_dim': 256,
                   'dec_dim': 256,
                   'word_embed_dim': 50,
                   'rel_embed_dim': 64,
                   'pos_embed_dim': 8,
                   'latent_dim': 64,
                   'input_dropout': trial.suggest_discrete_uniform('input_dropout', 0.0, 0.8, 0.05),
                   'output_dropout': trial.suggest_discrete_uniform('output_dropout', 0.0, 0.5, 0.05),
                   'weight_decay': trial.suggest_categorical('weight_decay', [0.0, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]),
                   'lr': 0.001,
                   'clip': trial.suggest_categorical('clip', [0, 1, 5, 10]),
                   'batch_size': 128,
                   'max_vocab_size': 40000 if model_name == 'nyt' else 50000,
                   'epochs': 100,
                   'device': 0,
                   'early_stop': True,
                   'patience': 5,
                   'runs': 3,
                   'max_sent_len': 50,
                   'show_example': False,
                   'primary_metric': 'pr_auc',
                   'bag_size': 500,
                   'teacher_force': trial.suggest_discrete_uniform('teacher_force', 0.2, 0.5, 0.05),
                   'enc_layers': 1,
                   'dec_layers': 1,
                   'enc_bidirectional': True,
                   'dec_bidirectional': False,
                   'optimizer': 'Adam',
                   'include_positions': True,
                   'lowercase': True,
                   'priors': False,
                   'task_weight': trial.suggest_discrete_uniform('task_weight', 0.6, 0.9, 0.05),
                   'freeze_words': False,
                   'reconstruction': True,
                   'log_interval': 100,
                   'reco_loss': 'AdaptiveSoftmax'}
    return space_prior


def model_space_base(trial, model_name):
    space_base = {'train_data': train_data_nyt if model_name == 'nyt' else train_data_wiki,
                  'val_data': val_data_nyt if model_name == 'nyt' else val_data_wiki,
                  'test_data': test_data_nyt if model_name == 'nyt' else test_data_wiki,
                  'relations_file': rels_nyt if model_name == 'nyt' else rels_wiki,
                  'output_folder': '../saved_modes_tuning/',
                  'pretrained_embeds_file': '../embeds/glove.6B.50d.txt',
                  'enc_dim': 256,
                  'dec_dim': 0,
                  'word_embed_dim': 50,
                  'rel_embed_dim': 64,
                  'pos_embed_dim': 8,
                  'input_dropout': trial.suggest_discrete_uniform('input_dropout', 0.0, 1.0, 0.05),
                  'output_dropout': trial.suggest_discrete_uniform('output_dropout', 0.0, 1.0, 0.05),
                  'weight_decay': trial.suggest_categorical('weight_decay', [0.0, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]),
                  'lr': 0.001,
                  'clip': trial.suggest_categorical('clip', [0, 1, 5, 10]),
                  'batch_size': 128,
                  'latent_dim': 0,
                  'max_vocab_size': 40000 if model_name == 'nyt' else 50000,
                  'epochs': 100,
                  'device': 0,
                  'early_stop': True,
                  'patience': 5,
                  'runs': 3,
                  'max_sent_len': 50,
                  'show_example': False,
                  'primary_metric': 'pr_auc',
                  'bag_size': 500,
                  'enc_layers': 1,
                  'dec_layers': 1,
                  'enc_bidirectional': True,
                  'dec_bidirectional': False,
                  'optimizer': 'Adam',
                  'include_positions': True,
                  'lowercase': True,
                  'priors': False,
                  'freeze_words': False,
                  'reconstruction': False,
                  'log_interval': 100,
                  'task_weight': 0,
                  'teacher_force': 0,
                  'reco_loss': 'AdaptiveSoftmax'}
    return space_base



