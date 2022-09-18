#!/usr/bin/env python3

"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Experiment Portal.
"""

import copy
import itertools
from pyexpat.errors import XML_ERROR_DUPLICATE_ATTRIBUTE
import string
from tkinter import Scrollbar
import numpy as np
import os, sys
import random
from datetime import datetime
from pprint import pprint
import logging

import torch

from src.parse_args import parser
from src.parse_args import args
import src.data_utils as data_utils
import src.eval
from src.hyperparameter_range import hp_range
from src.knowledge_graph import KnowledgeGraph
from src.emb.fact_network import ComplEx, ConvE, DistMult
from src.emb.fact_network import get_conve_kg_state_dict, get_complex_kg_state_dict, get_distmult_kg_state_dict
from src.emb.emb import EmbeddingBasedMethod
from src.rl.graph_search.pn import GraphSearchPolicy
from src.rl.graph_search.pg import PolicyGradient
from src.rl.graph_search.rs_pg import RewardShapingPolicyGradient
from src.utils.ops import flatten
logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

torch.cuda.set_device(args.gpu)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def process_data():
    data_dir = args.data_dir
    raw_kb_path = os.path.join(data_dir, 'raw.kb')
    train_path = data_utils.get_train_path(args)
    dev_path = os.path.join(data_dir, 'dev.triples')
    test_path = os.path.join(data_dir, 'test.triples')
    data_utils.prepare_kb_envrioment(raw_kb_path, train_path, dev_path, test_path, args.test, args.add_reverse_relations)

def construct_model(args, logger):
    """
    Construct NN graph.
    """
    kg = KnowledgeGraph(args)
    if args.model.endswith('.gc'):
        kg.load_fuzzy_facts()

    logger.info('Total number of entities {}'.format(len(kg.entity2id))) # PLACEHOLDER NUMBER
    logger.info('Total number of relations {}'.format(len(kg.relation2id))) # PLACEHOLDER NUMBER

    if args.model.startswith('point.rs'):
        pn = GraphSearchPolicy(args)
        fn_model = args.model.split('.')[2]
        fn_args = copy.deepcopy(args)
        fn_args.model = fn_model
        fn_args.relation_only = False
        fn = ConvE(fn_args, kg.num_entities)
        fn_kg = KnowledgeGraph(fn_args)
        lf = RewardShapingPolicyGradient(args, kg, pn, fn_kg, fn, logger)
    elif args.model == 'conve':
        fn = ConvE(args, kg.num_entities)
        lf = EmbeddingBasedMethod(args, kg, fn)
    else:
        raise NotImplementedError
    return lf

def train(lf):
    train_path = data_utils.get_train_path(args)
    dev_path = os.path.join(args.data_dir, 'dev.triples')
    entity_index_path = os.path.join(args.data_dir, 'entity2id.txt')
    relation_index_path = os.path.join(args.data_dir, 'relation2id.txt')
    train_data = data_utils.load_triples(
        train_path, entity_index_path, relation_index_path, group_examples_by_query=args.group_examples_by_query,
        add_reverse_relations=args.add_reversed_training_edges)
    if 'NELL' in args.data_dir:
        adj_list_path = os.path.join(args.data_dir, 'adj_list.pkl')
        seen_entities = data_utils.load_seen_entities(adj_list_path, entity_index_path)
    else:
        seen_entities = set()
    dev_data = data_utils.load_triples(dev_path, entity_index_path, relation_index_path, seen_entities=seen_entities)
    if args.checkpoint_path is not None:
        lf.load_checkpoint(args.checkpoint_path)
    lf.run_train(train_data, dev_data)

def inference_sl(lf, orig_args):
    lf.batch_size = orig_args.dev_batch_size
    lf.eval()
    entity_index_path = os.path.join(orig_args.data_dir, 'entity2id.txt')
    relation_index_path = os.path.join(orig_args.data_dir, 'relation2id.txt')
    seen_entities = set()

    eval_metrics = {
        'dev': {},
        'test': {}
    }
    # evaluating by relation type
    dev_path = os.path.join(orig_args.data_dir, 'dev.triples')
    dev_data = data_utils.load_triples(dev_path, entity_index_path, relation_index_path, seen_entities=seen_entities)
    pred_scores = lf.forward(dev_data, verbose=False)
    to_m_rels, to_1_rels, _ = data_utils.get_relations_by_type(orig_args.data_dir, relation_index_path)
    relation_by_types = (to_m_rels, to_1_rels)

    print('Dev set evaluation by relation type (partial graph)')
    to_m_mrr, to_1_mrr = src.eval.hits_and_ranks_by_relation_type(dev_data, pred_scores, lf.kg.dev_objects, relation_by_types, verbose=True)
    print('Dev set evaluation by relation type (full graph)')
    to_m_mrr, to_1_mrr = src.eval.hits_and_ranks_by_relation_type(dev_data, pred_scores, lf.kg.all_objects, relation_by_types, verbose=True)

    with open(orig_args.checkpoint_dir + "full_scores.txt", 'a+') as scoresfile:
        scoresfile.write("Dev set evaluation by relation type (partial graph)\nto many: {}\nto one: {}\n\n".format(to_m_mrr, to_1_mrr))
        scoresfile.write("Dev set evaluation by relation type (full graph)\nto many: {}\nto one: {}\n\n".format(to_m_mrr, to_1_mrr))

    # plain evaluation on test and dev sets
    test_path = os.path.join(orig_args.data_dir, 'test.triples')
    test_data = data_utils.load_triples(
        test_path, entity_index_path, relation_index_path, seen_entities=seen_entities, verbose=False)
    pred_scores = lf.forward(test_data, verbose=False)
    test_metrics = src.eval.hits_and_ranks(test_data, pred_scores, lf.kg.all_objects, verbose=True)
    eval_metrics['test']['hits_at_1'] = test_metrics[0]
    eval_metrics['test']['hits_at_3'] = test_metrics[1]
    eval_metrics['test']['hits_at_5'] = test_metrics[2]
    eval_metrics['test']['hits_at_10'] = test_metrics[3]
    eval_metrics['test']['mrr'] = test_metrics[4]
    with open(orig_args.checkpoint_dir + "scores.txt", 'a+') as scoresfile:
        print("writing to scores file")
        scoresfile.write("End of RL performance performance:\n"
                        + "Hits@1:  {:.4f}\nHits@3:  {:.4f}\nHits@5:  {:.4f}\nHits@10:  {:.4f}\nMRR:  {:.4f}\n\n".format(*test_metrics))

def create_sl_checkpoint(checkpoint, orig_args):
    original_model_dir = orig_args.training_state_dicts

    # make checkpoint folder
    orig_args.checkpoint_dir = orig_args.model_dir + '/checkpoint_sl_epoch_{}/'.format(checkpoint)
    os.mkdir(orig_args.checkpoint_dir)

    # make model folder
    os.mkdir(orig_args.checkpoint_dir + '/model_weights/')
    orig_args.checkpoint_model_dir = orig_args.checkpoint_dir + '/model_weights/'

    # make trainer
    orig_args.sl = False
    lf_rl = construct_model(orig_args)
    lf_rl.cuda()
    lf_rl.load_state_dict(torch.load(original_model_dir + "model_checkpoint.pth"))

    print("training RL")
    train(lf_rl)

    print("testing RL")
    with open(orig_args.checkpoint_model_dir + '/scores.txt', 'a') as score_file:
        score_file.write("Final score: ")
    inference_sl(lf_rl, orig_args)

    print("saving state")
    torch.save(lf_rl.state_dict(), orig_args.checkpoint_model_dir + "model_checkpoint.pth")

def run_experiment(args):

    # if args.test:
    #     if 'NELL' in args.data_dir:
    #         dataset = os.path.basename(args.data_dir)
    #         args.distmult_state_dict_path = data_utils.change_to_test_model_path(dataset, args.distmult_state_dict_path)
    #         args.complex_state_dict_path = data_utils.change_to_test_model_path(dataset, args.complex_state_dict_path)
    #         args.conve_state_dict_path = data_utils.change_to_test_model_path(dataset, args.conve_state_dict_path)
    #     args.data_dir += '.test'

    if args.process_data:
        # Process knowledge graph data
        process_data()
    else:
        with torch.set_grad_enabled(args.train or args.search_random_seed or args.grid_search):

            model_root_dir = args.model_root_dir
            model_sub_dir = args.experiment_name + "_" + datetime.now().strftime("%m%d%Y_%H%M%S")
            model_dir = os.path.join(model_root_dir, model_sub_dir)

            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
                print('Model directory created: {}'.format(model_dir))
            else:
                print('Model directory exists: {}'.format(model_dir))

            args.model_dir = model_dir
            args.training_state_dicts = args.model_dir + "/model/"
            os.mkdir(args.training_state_dicts)
            args.log_file_name = args.model_dir + '/log.txt'

            with open(args.model_dir + '/config.txt', 'w') as out:
                pprint(args, stream=out)

            original_args = copy.deepcopy(args)

            # Set logging
            logger.setLevel(logging.INFO)
            fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                                    '%m/%d/%Y %I:%M:%S %p')
            console = logging.StreamHandler()
            console.setFormatter(fmt)
            logger.addHandler(console)
            logfile = logging.FileHandler(args.log_file_name, 'w')
            logfile.setFormatter(fmt)
            logger.addHandler(logfile)

            # Create the SL trainer
            args.sl = True
            args.learning_rate = args.learning_rate_sl
            args.checkpoint_path = None
            args.total_iterations = args.total_iterations_sl
            lf = construct_model(args, logger)
            lf.cuda()

            # Create checkpoint for pure RL run
            last_epoch = 0
            torch.save(lf.state_dict(), args.training_state_dicts + "model_checkpoint.pth")
            create_sl_checkpoint(last_epoch, copy.deepcopy(original_args))
            lf.load_state_dict(torch.load(args.training_state_dicts + "model_checkpoint.pth"))

            for ckpt in range(1, original_args.sl_checkpoints):
                train(lf)

                torch.save(lf.state_dict(), args.training_state_dicts + "model_checkpoint.pth")
                create_sl_checkpoint(ckpt, copy.deepcopy(original_args))
                lf.load_state_dict(torch.load(args.training_state_dicts + "model_checkpoint.pth"))

if __name__ == '__main__':
    run_experiment(args)
