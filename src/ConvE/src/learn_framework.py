"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Base learning framework.
"""

import os
import random
import shutil
import sys
from tkinter import E
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

import src.eval
from src.utils.ops import var_cuda, zeros_var_cuda
import src.utils.ops as ops


class LFramework(nn.Module):
    def __init__(self, args, kg, mdl, logger):
        super(LFramework, self).__init__()
        self.args = args
        self.data_dir = args.data_dir
        self.model_dir = args.model_dir
        self.model = args.model
        self.supervised_learning_mode = args.sl

        # logging
        self.logger = logger

        # Training hyperparameters
        self.batch_size = args.batch_size
        self.train_batch_size = args.train_batch_size
        self.dev_batch_size = args.dev_batch_size
        self.start_epoch = args.start_epoch
        self.num_epochs = args.num_epochs
        self.num_wait_epochs = args.num_wait_epochs
        self.num_peek_epochs = args.num_peek_epochs
        self.learning_rate = args.learning_rate
        self.grad_norm = args.grad_norm
        self.adam_beta1 = args.adam_beta1
        self.adam_beta2 = args.adam_beta2
        self.optim = None

        self.inference = not args.train
        self.run_analysis = args.run_analysis

        self.kg = kg
        self.mdl = mdl
        print('{} module created'.format(self.model))

    def print_all_model_parameters(self):
        print('\nModel Parameters')
        print('--------------------------')
        for name, param in self.named_parameters():
            print(name, param.numel(), 'requires_grad={}'.format(param.requires_grad))
        param_sizes = [param.numel() for param in self.parameters()]
        print('Total # parameters = {}'.format(sum(param_sizes)))
        print('--------------------------')
        print()

    def run_train(self, train_data, dev_data):
        self.print_all_model_parameters()

        if self.optim is None:
            self.optim = optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)

        # Track dev metrics changes
        best_dev_metrics = 0
        dev_metrics_history = []
        batch_count = 0
        epoch_id = 0

        while batch_count < self.args.total_iterations:
            epoch_id += 1

            if self.rl_variation_tag.startswith('rs'):
                # Reward shaping module sanity check:
                #   Make sure the reward shaping module output value is in the correct range
                train_scores = self.test_fn(train_data)
                dev_scores = self.test_fn(dev_data)
                print('Train set average fact score: {}'.format(float(train_scores.mean())))
                print('Dev set average fact score: {}'.format(float(dev_scores.mean())))

            # puts model in train mode
            self.train()

            if self.rl_variation_tag.startswith('rs'):
                self.fn.eval()
                self.fn_kg.eval()
                if self.model.endswith('hypere'):
                    self.fn_secondary_kg.eval()

            self.batch_size = self.train_batch_size
            random.shuffle(train_data)
            
            for example_id in tqdm(range(0, len(train_data), self.batch_size)):
                batch_count += 1
                self.optim.zero_grad()

                mini_batch = train_data[example_id:example_id + self.batch_size]
                if len(mini_batch) < self.batch_size:
                    continue
                loss = self.loss(mini_batch)
                loss['model_loss'].backward()
                if self.grad_norm > 0:
                    clip_grad_norm_(self.parameters(), self.grad_norm)

                self.optim.step()

                if self.args.model == 'conve':
                    outstr = "ConvE, epoch count: {1:4d}, overall batch count: {1:4d}, loss: {2:7.4f}".format(epoch_id, batch_count, loss['print_loss'])
                else:
                    reward_reshape = np.reshape(loss['reward'], (self.batch_size, self.args.num_rollouts))
                    reward_reshape = np.sum(reward_reshape, axis=1)
                    reward_reshape = (reward_reshape > 0)
                    num_ep_correct = np.sum(reward_reshape)

                    outstr = "RL: {0:4d}, epoch count: {1:4d}, overall batch count: {1:4d}, num_hits: {2:7.4f}, avg. reward per batch {3:7.4f}, "+\
                            "num_ep_correct {4:4d}, avg_ep_correct {5:7.4f}, train loss {6:7.4f}".format(int(self.supervised_learning_mode), epoch_id, batch_count, np.sum(loss['reward']), np.mean(loss['reward']), num_ep_correct,
                                    (num_ep_correct / self.batch_size),
                                    loss['print_loss'])
                self.logger.info(outstr)  
                print(outstr)

                if batch_count >= self.args.total_iterations:
                    break

            # Check in-progress scores for SL portion of RL+SL training
            if epoch_id > 0 and epoch_id % self.num_peek_epochs == 0 and (self.supervised_learning_mode or self.args.model == 'conve'):

                self.eval()
                self.batch_size = self.dev_batch_size
                dev_scores = self.forward(dev_data, verbose=False)
                scoresvalues = src.eval.hits_and_ranks(dev_data, dev_scores, self.kg.dev_objects, verbose=False)
                metrics = scoresvalues[-1]

                print('Dev set performance: (include test set labels)')
                in_progress_scores = src.eval.hits_and_ranks(dev_data, dev_scores, self.kg.all_objects, verbose=True)
                
                if self.model.startswith('point'):

                    # write in progress scores to scores file
                    with open(self.args.checkpoint_dir + "scores.txt", 'a+') as scoresfile:
                        print("writing to scores file")
                        scoresfile.write("In progress score epoch {} batch {}:\n".format(epoch_id, batch_count)
                                        + "Hits@1:  {:.4f}\nHits@3:  {:.4f}\nHits@5:  {:.4f}\nHits@10:  {:.4f}\nMRR:  {:.4f}\n\n".format(*in_progress_scores))

                    # Action dropout annealing
                    eta = self.action_dropout_anneal_interval
                    if len(dev_metrics_history) > eta and metrics < min(dev_metrics_history[-eta:]):
                        old_action_dropout_rate = self.action_dropout_rate
                        self.action_dropout_rate *= self.action_dropout_anneal_factor 
                        print('Decreasing action dropout rate: {} -> {}'.format(
                            old_action_dropout_rate, self.action_dropout_rate))

                else:
                    # save best conve model
                    if metrics > best_dev_metrics:
                        self.save_checkpoint(checkpoint_id=epoch_id, epoch_id=epoch_id)
                        best_dev_metrics = metrics
                        with open(os.path.join(self.model_dir, 'best_dev_iteration.dat'), 'w') as o_f:
                            o_f.write('{}'.format(epoch_id))


    def forward(self, examples, verbose=False):

        pred_scores = []

        for example_id in tqdm(range(0, len(examples), self.batch_size)):

            mini_batch = examples[example_id:example_id + self.batch_size]
            mini_batch_size = len(mini_batch)

            if len(mini_batch) < self.batch_size:
                self.make_full_batch(mini_batch, self.batch_size)

            pred_score = self.predict(mini_batch, verbose=verbose)
            pred_scores.append(pred_score[:mini_batch_size])

        scores = torch.cat(pred_scores)
        return scores

    def format_batch(self, batch_data, num_labels=-1, num_tiles=1):
        """
        Convert batched tuples to the tensors accepted by the NN.
        """
        def convert_to_binary_multi_subject(e1):
            e1_label = zeros_var_cuda([len(e1), num_labels])
            for i in range(len(e1)):
                e1_label[i][e1[i]] = 1
            return e1_label

        def convert_to_binary_multi_object(e2):
            e2_label = zeros_var_cuda([len(e2), num_labels])
            for i in range(len(e2)):
                e2_label[i][e2[i]] = 1
            return e2_label

        batch_e1, batch_e2, batch_r = [], [], []
        for i in range(len(batch_data)):
            e1, e2, r = batch_data[i]
            batch_e1.append(e1)
            batch_e2.append(e2)
            batch_r.append(r)
        batch_e1 = var_cuda(torch.LongTensor(batch_e1), requires_grad=False)
        batch_r = var_cuda(torch.LongTensor(batch_r), requires_grad=False)
        if type(batch_e2[0]) is list:
            batch_e2 = convert_to_binary_multi_object(batch_e2)
        elif type(batch_e1[0]) is list:
            batch_e1 = convert_to_binary_multi_subject(batch_e1)
        else:
            batch_e2 = var_cuda(torch.LongTensor(batch_e2), requires_grad=False)

        if num_tiles > 1:
            batch_e1 = ops.tile_along_beam(batch_e1, num_tiles)
            batch_r = ops.tile_along_beam(batch_r, num_tiles)
            batch_e2 = ops.tile_along_beam(batch_e2, num_tiles)
        
        return batch_e1, batch_e2, batch_r

    def make_full_batch(self, mini_batch, batch_size, multi_answers=False):
        dummy_e = self.kg.dummy_e
        dummy_r = self.kg.dummy_r
        if multi_answers:
            dummy_example = (dummy_e, [dummy_e], dummy_r)
        else:
            dummy_example = (dummy_e, dummy_e, dummy_r)
        for _ in range(batch_size - len(mini_batch)):
            mini_batch.append(dummy_example)

    def save_checkpoint(self, checkpoint_id, epoch_id=None):
        """
        Save a new best conve model checkpoint.
        :param checkpoint_id: Model checkpoint index assigned by training loop.
        :param epoch_id: Model epoch index assigned by training loop.
        :param is_best: if set, the model being saved is the best model on dev set.
        """
        checkpoint_dict = dict()
        checkpoint_dict['state_dict'] = self.state_dict()
        checkpoint_dict['epoch_id'] = epoch_id

        out_tar = os.path.join(self.model_dir, 'checkpoint-{}.tar'.format(checkpoint_id))
        best_path = os.path.join(self.model_dir, 'model_best.tar')
        shutil.copyfile(out_tar, best_path)
        print('=> best model updated \'{}\''.format(best_path))

    def load_checkpoint(self, input_file):
        """
        Load model checkpoint.
        :param n: Neural network module.
        :param self.kg: Knowledge graph module.
        :param input_file: Checkpoint file path.
        """
        if os.path.isfile(input_file):
            print('=> loading checkpoint \'{}\''.format(input_file))
            checkpoint = torch.load(input_file, map_location="cuda:{}".format(self.args.gpu))
            self.load_state_dict(checkpoint['state_dict'])
            if not self.inference:
                self.start_epoch = checkpoint['epoch_id'] + 1
                assert (self.start_epoch <= self.num_epochs)
        else:
            print('=> no checkpoint found at \'{}\''.format(input_file))

    def export_to_embedding_projector(self):
        """
        Export knowledge base embeddings into .tsv files accepted by the Tensorflow Embedding Projector.
        """
        vector_path = os.path.join(self.model_dir, 'vector.tsv')
        meta_data_path = os.path.join(self.model_dir, 'metadata.tsv')
        v_o_f = open(vector_path, 'w')
        m_o_f = open(meta_data_path, 'w')
        for r in self.kg.relation2id:
            if r.endswith('_inv'):
                continue
            r_id = self.kg.relation2id[r]
            R = self.kg.relation_embeddings.weight[r_id]
            r_print = ''
            for i in range(len(R)):
                r_print += '{}\t'.format(float(R[i]))
            v_o_f.write('{}\n'.format(r_print.strip()))
            m_o_f.write('{}\n'.format(r))
            print(r, '{}'.format(float(R.norm())))
        v_o_f.close()
        m_o_f.close()
        print('KG embeddings exported to {}'.format(vector_path))
        print('KG meta data exported to {}'.format(meta_data_path))

    @property
    def rl_variation_tag(self):
        parts = self.model.split('.')
        if len(parts) > 1:
            return parts[1]
        else:
            return ''
