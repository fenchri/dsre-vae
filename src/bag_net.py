#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 04/03/2020

author: fenia
"""

import torch
from torch import nn, torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from init_net import BaseNet
import numpy as np


class BagNet(BaseNet):
    """
    Model architecture
    """
    def greedy_decoding(self, z):
        h0 = self.latent2hid(z).unsqueeze(0)
        h0 = h0.expand(self.config['dec_layers'], h0.size(1), h0.size(2)).contiguous()
        c0 = torch.zeros(self.config['dec_layers'], z.size(0), self.config['dec_dim']).to(self.device).contiguous()

        # start with start-of-sentence (SOS)
        w_id = torch.empty((1,)).fill_(self.w_vocab.word2id[self.w_vocab.SOS]).to(self.device).long()
        gen_sentence = [self.w_vocab.SOS]

        while (gen_sentence[-1] != self.w_vocab.EOS) and (len(gen_sentence) <= self.config['max_sent_len']):
            dec_input = self.w_embed(w_id)
            dec_input = torch.cat([dec_input.unsqueeze(0), z.unsqueeze(0)], dim=2)

            next_word_rep, (h0, c0) = self.lang_decoder(dec_input, hidden_=(h0, c0))

            logits = self.reco_loss.log_prob(next_word_rep.squeeze(0))
            norm_logits = F.softmax(logits, dim=1)

            # w_id = torch.multinomial(norm_logits.squeeze(0), 1)
            w_id = norm_logits.argmax(dim=1)
            gen_sentence += [self.w_vocab.id2word[w_id.item()]]

        gen_sentence = ' '.join(gen_sentence[1:-1])
        print(gen_sentence + '\n')

    def sample_posterior(self, batch):
        x_vec = self.w_embed(batch['source'])  # (all-batch-sents, words, dim)

        if self.config['include_positions']:
            pos1 = self.p_embed(batch['pos1'])
            pos2 = self.p_embed(batch['pos2'])
            x_vec = torch.cat([x_vec, pos1, pos2], dim=2)
        x_vec = self.in_drop(x_vec)

        enc_out, (hidden, cell_state) = self.lang_encoder(x_vec, len_=batch['sent_len'])  # encoder

        new_input = torch.cat([hidden, cell_state], dim=1)
        mu_ = self.hid2mu(new_input)  # use sentence representation for reconstruction
        logvar_ = self.hid2var(new_input)
        latent_z1 = self.reparameterisation(mu_, logvar_)
        latent_z2 = self.reparameterisation(mu_, logvar_)
        latent_z3 = self.reparameterisation(mu_, logvar_)

        names = list(batch['bag_names'])
        r = np.repeat(np.arange(len(names)), repeats=batch['bag_size'].cpu().tolist())

        for i, (m, z1, z2, z3) in enumerate(zip(mu_, latent_z1, latent_z2, latent_z3)):  # for each sentence
            all_w_ids = [self.w_vocab.id2word[w_.item()] for w_ in batch['source'][i] if w_.item() != 0]

            arg1 = ' '.join(all_w_ids[batch['mentions'][i][0]:batch['mentions'][i][1] + 1])
            arg2 = ' '.join(all_w_ids[batch['mentions'][i][2]:batch['mentions'][i][3] + 1])

            if len(all_w_ids) <= 20 and ('NA' != names[r[i]].split(' ### ')[2]):
                print(' '.join(all_w_ids))
                print(arg1, '#', arg2, '#', names[r[i]])
                print('=' * 50)
                print('MEAN = ', end='')
                self.greedy_decoding(m.unsqueeze(0))
                print('SAMPLE 1 = ', end='')
                self.greedy_decoding(z1.unsqueeze(0))
                print('SAMPLE 2 = ', end='')
                self.greedy_decoding(z2.unsqueeze(0))
                print('SAMPLE 3 = ', end='')
                self.greedy_decoding(z3.unsqueeze(0))
                print('-' * 50)

    def homotomies(self):
        print(10 * '=', 'Generating homotomies', 10 * '=')

        z1 = torch.randn([1, self.config['latent_dim']]).to(self.device)  # random sample
        z2 = torch.randn([1, self.config['latent_dim']]).to(self.device)
        z = 0.2 * z1 + 0.8 * z2

        self.greedy_decoding(z1)
        for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            z = i * z1 + (1 - i) * z2
            self.greedy_decoding(z)
        self.greedy_decoding(z2)

    def generation(self, examples_num=10):
        """
        Generate sentences given a random sample
        """
        print(10 * '=', 'Generating sentences', 10 * '=')
        for i in range(examples_num):
            z = torch.randn([1, self.config['latent_dim']]).to(self.device)
            self.greedy_decoding(z)

    # ----------------------------------------------- #

    def reparameterisation(self, mean_, logvar_):
        std = torch.exp(0.5 * logvar_)
        eps = torch.randn_like(std)
        return mean_ + (eps * std)

    def merge_tokens(self, enc_seq, mentions):
        """
        Merge tokens into mentions;
        Find which tokens belong to a mention (based on start-end ids) and average them
        """
        start1, end1, w_ids1 = torch.broadcast_tensors(mentions[:, 0].unsqueeze(-1),
                                                       mentions[:, 1].unsqueeze(-1),
                                                       torch.arange(0, enc_seq.shape[1]).unsqueeze(0).to(self.device))

        start2, end2, w_ids2 = torch.broadcast_tensors(mentions[:, 2].unsqueeze(-1),
                                                       mentions[:, 3].unsqueeze(-1),
                                                       torch.arange(0, enc_seq.shape[1]).unsqueeze(0).to(self.device))

        index_t1 = (torch.ge(w_ids1, start1) & torch.le(w_ids1, end1)).float().to(self.device).unsqueeze(1)
        index_t2 = (torch.ge(w_ids2, start2) & torch.le(w_ids2, end2)).float().to(self.device).unsqueeze(1)

        arg1 = torch.div(torch.matmul(index_t1, enc_seq), torch.sum(index_t1, dim=2).unsqueeze(-1)).squeeze(1)  # avg
        arg2 = torch.div(torch.matmul(index_t2, enc_seq), torch.sum(index_t2, dim=2).unsqueeze(-1)).squeeze(1)  # avg
        return arg1, arg2

    def form_decoder_input(self, words):
        """ Word dropout: https://www.aclweb.org/anthology/K16-1002/ """

        random_z2o = torch.rand(words.size()).to(self.device)
        cond1 = torch.lt(random_z2o, self.config['teacher_force'])  # if < word_drop
        cond2 = torch.ne(words, self.PAD_id) & \
                torch.ne(words, self.SOS_id)  # if != PAD & SOS

        dec_input = torch.where(cond1 & cond2,
                                torch.full_like(words, self.UNK_id),
                                words)
        dec_input = self.w_embed(dec_input)
        return dec_input

    def reconstruction(self, latent_z, batch):
        y_vec = self.form_decoder_input(batch['source'])
        y_vec = torch.cat([y_vec,
                           latent_z.unsqueeze(dim=1).repeat((1, y_vec.size(1), 1))], dim=2)

        h_0 = self.latent2hid(latent_z).unsqueeze(0)
        h_0 = h_0.expand(self.config['dec_layers'], h_0.size(1), h_0.size(2))
        c_0 = torch.zeros(self.config['dec_layers'], latent_z.size(0), self.config['dec_dim']).to(self.device)

        recon_x, _ = self.lang_decoder(y_vec, len_=batch['sent_len'], hidden_=(h_0, c_0))
        return recon_x

    def calc_reconstruction_loss(self, recon_x, batch):
        # remove padded
        tmp = torch.arange(recon_x.size(1)).repeat(batch['sent_len'].size(0), 1).to(self.device)
        mask = torch.lt(tmp, batch['sent_len'][:, None].repeat(1, tmp.size(1)))  # result in (words, dim)

        # Convert to (sentences, words)
        o_vec = self.reco_loss(recon_x[mask], batch['target'][mask])  # (words,)
        o_vec = pad_sequence(torch.split(o_vec.loss, batch['sent_len'].tolist(), dim=0),
                             batch_first=True,
                             padding_value=0)

        mean_mean = torch.div(torch.sum(o_vec, dim=1), batch['sent_len'].float().to(self.device))
        reco_loss = {'mean': torch.mean(mean_mean),  # mean over words, mean over batch (for perplexity)
                     'sum': torch.mean(torch.sum(o_vec, dim=1))}  # sum over words, mean over batch (sentences)
        return reco_loss

    @staticmethod
    def calc_kld(mu, logvar, mu_prior=None, logvar_prior=None):
        if mu_prior is not None:
            mu_diff = mu_prior.float() - mu
            kld = -0.5 * (1 + logvar - mu_diff.pow(2) - logvar.exp())
        else:
            kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

        kld = torch.sum(kld, dim=1)
        kld = torch.mean(kld)  # sum over dim, mean over batch
        return kld

    def calc_task_loss(self, rel_logits, target):
        task_loss = self.task_loss(rel_logits, target.type_as(rel_logits))  # (examples, categories)
        task_loss = torch.sum(task_loss, dim=1)
        task_loss = torch.mean(task_loss)  # sum across relations, mean over batch
        rel_probs = torch.sigmoid(rel_logits)
        return rel_probs, task_loss

    def forward(self, batch):
        ####################
        # Encoder
        ####################
        x_vec = self.w_embed(batch['source'])     # (all-batch-sents, words, dim)

        if self.config['include_positions']:
            pos1 = self.p_embed(batch['pos1'])
            pos2 = self.p_embed(batch['pos2'])
            x_vec = torch.cat([x_vec, pos1, pos2], dim=2)
        x_vec = self.in_drop(x_vec)

        enc_out, (hidden, cell_state) = self.lang_encoder(x_vec, len_=batch['sent_len'])  # encoder

        ##########################
        # Argument Representation
        ##########################
        arg1, arg2 = self.merge_tokens(enc_out, batch['mentions'])  # contextualised representations of arguments

        #####################
        # Reconstruction
        #####################
        if self.config['reconstruction']:
            new_input = torch.cat([hidden, cell_state], dim=1)

            # create hidden code
            mu_ = self.hid2mu(new_input)
            logvar_ = self.hid2var(new_input)
            latent_z = self.reparameterisation(mu_, logvar_)

            if self.config['priors']:
                prior_mus_expanded = torch.repeat_interleave(batch['prior_mus'], repeats=batch['bag_size'], dim=0)
                kld = self.calc_kld(mu_, logvar_, mu_prior=prior_mus_expanded)

            else:
                kld = self.calc_kld(mu_, logvar_)

            # reconstruction
            recon_x = self.reconstruction(latent_z, batch)
            reco_loss = self.calc_reconstruction_loss(recon_x, batch)

            # sentence representation --> use info from VAE !!
            sent_rep = torch.cat([latent_z, arg1, arg2], dim=1)

        else:
            kld = torch.zeros((1,)).to(self.device)
            reco_loss = {'sum': torch.zeros((1,)).to(self.device),
                         'mean': torch.zeros((1,)).to(self.device)}
            mu_ = torch.zeros((hidden.size(0), self.config['latent_dim'])).to(self.device)

            # sentence representation
            sent_rep = torch.cat([hidden, arg1, arg2], dim=1)

        # Sentences per bag
        sent_rep = pad_sequence(torch.split(sent_rep, batch['bag_size'].tolist(), dim=0),
                                batch_first=True,
                                padding_value=0)

        # Sentence-level ATTENTION: select sentences from each bag
        sent_rep = self.reduction(sent_rep)
        sent_rep = self.sentence_attention(sent_rep, batch['bag_size'], self.r_embed.embedding.weight.data)

        ####################
        # Classification
        ####################
        sent_rep = self.out_drop(sent_rep)
        sent_rep = self.dim2rel(sent_rep)   # tie embeds
        sent_rep = sent_rep.diagonal(dim1=1, dim2=2)  # take probs based on relation query vector
        rel_probs, task_loss = self.calc_task_loss(sent_rep, batch['rel'])
        assert torch.sum(torch.isnan(rel_probs)) == 0.0, sent_rep

        return task_loss, rel_probs, kld, reco_loss, mu_
