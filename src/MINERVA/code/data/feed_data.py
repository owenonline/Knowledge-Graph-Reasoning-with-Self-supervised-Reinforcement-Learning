from tqdm import tqdm
import json
import numpy as np
from collections import defaultdict
import csv
import random
import os

class RelationEntityBatcher():
    def __init__(self, rl, orig, dataset,batch_size, entity_vocab, relation_vocab, mode = "train", num_rollouts=20):
        # self.input_dir = input_dir
        # self.input_file = input_dir+'/{0}.txt'.format(mode)
        self.rl = rl
        self.orig = orig
        self.train_data = dataset['train']
        self.test_data = dataset['test']
        self.dev_data = dataset['dev']
        self.graph_data = dataset['graph'] 
        self.batch_size = batch_size
        self.full_graph = dataset['full_graph']
        self.num_rollouts = num_rollouts
        print('Reading vocab...')
        self.entity_vocab = entity_vocab
        self.relation_vocab = relation_vocab
        self.mode = mode
        self.relation_counts = defaultdict(int)
        self.create_triple_store()
        
        # if self.rl:
        #     self.create_triple_store()
        # else:
        #     self.create_sl_triple_store()
        # print("batcher loaded")

    # get a training set with an even number of queries of each relation type
    def create_sl_triple_store(self):
        self.store_all_correct = defaultdict(set)
        self.store = []

        self.rstore = defaultdict(list)

        for line in self.train_data:
            e1 = self.entity_vocab[line[0]]
            r = self.relation_vocab[line[1]]
            e2 = self.entity_vocab[line[2]]

            self.rstore[r].append([e1,r,e2])

        type_count = min([len(self.rstore[key]) for key in self.rstore])

        for key in self.rstore:
            selection = random.choice(self.rstore[key], type_count)
            for query in selection:
                e1 = query[0]
                r = query[1]
                e2 = query[2]

                self.store.append([e1,r,e2])
                self.store_all_correct[(e1, r)].add(e2)  #YM: there may exist multiple answers for the same query, i.e., same (e1,r) may mapping to different e2. store_all_correct will give all solution for the same query
        
        self.store = np.array(self.store)
        # self.store = np.array(self.store[:500])
        self.train_set_length = self.store.shape[0]
        print("Training set length is {}".format(self.train_set_length))
        print("Using {} of each relation type".format(type_count))

    def create_triple_store(self):
        self.store_all_correct = defaultdict(set)
        self.store = []
        
        if self.mode == 'train':
            # with open(input_file) as raw_input_file:
                # csv_file = csv.reader(raw_input_file, delimiter = '\t' )
            for line in self.train_data:
                e1 = self.entity_vocab[line[0]]
                r = self.relation_vocab[line[1]]
                e2 = self.entity_vocab[line[2]]
                self.relation_counts[r] += 1
                # if self.rl:
                #     e1 = self.entity_vocab[line[0]]
                #     r = self.relation_vocab[line[1]]
                #     e2 = self.entity_vocab[line[2]]
                # else:
                #     # this is because we store the embedding during the RL train step, so we don't have to fetch it again
                #     e1 = line[0]
                #     r = line[1]
                #     e2 = line[2]
                self.store.append([e1,r,e2])
                self.store_all_correct[(e1, r)].add(e2)  #YM: there may exist multiple answers for the same query, i.e., same (e1,r) may mapping to different e2. store_all_correct will give all solution for the same query
            self.store = np.array(self.store)
            # self.store = np.array(self.store[:500])
            self.train_set_length = self.store.shape[0]
            print("Training set length is {}".format(self.train_set_length))
        else:
            if self.mode == 'test':
                dataset = self.test_data
            if self.mode == 'dev':
                dataset = self.dev_data
            for line in dataset:
                e1 = line[0]
                r = line[1]
                e2 = line[2]
                if e1 in self.entity_vocab and e2 in self.entity_vocab:
                    e1 = self.entity_vocab[e1]
                    r = self.relation_vocab[r]
                    e2 = self.entity_vocab[e2]
                    self.store.append([e1,r,e2])
            self.store = np.array(self.store)

            for line in self.full_graph:
                e1 = line[0]
                r = line[1]
                e2 = line[2]
                if e1 in self.entity_vocab and e2 in self.entity_vocab:
                    e1 = self.entity_vocab[e1]
                    r = self.relation_vocab[r]
                    e2 = self.entity_vocab[e2]
                    self.store_all_correct[(e1, r)].add(e2)

    def yield_next_batch_train(self, labeller):
        while True:
            batch_idx = np.random.randint(0, self.store.shape[0], size=self.batch_size)
            batch = self.store[batch_idx, :]
            e1 = batch[:,0]
            r = batch[:, 1]
            e2 = batch[:, 2]
            all_e2s = []

            for i in range(e1.shape[0]):
                all_e2s.append(self.store_all_correct[(e1[i], r[i])])

            #generates correct path numbers
            labels=[[],[],[]]
            for i in range(len(batch)):
                labeller.correct_path(batch[i])

            assert e1.shape[0] == e2.shape[0] == r.shape[0] == len(all_e2s)
            if self.rl:
                yield e1, r, e2, all_e2s
            else:
                #generates correct paths
                labels=[[],[],[]]
                for i in range(len(batch)):
                    correct = labeller.correct_path(batch[i])
                    #handle rollouts 
                    for i in range(self.num_rollouts):
                        labels[0].append(correct[0])
                        labels[1].append(correct[1])
                        labels[2].append(correct[2])
                print("{} paths not found out of {} ({:.0%})".format(labeller.no_paths_found, len(batch), labeller.no_paths_found/(len(batch))))
                labeller.no_paths_found = 0
                yield e1, r, e2, all_e2s, labels

    def yield_next_batch_test(self):
        remaining_triples = self.store.shape[0]
        current_idx = 0
        while True:
            if remaining_triples == 0:
                return


            if remaining_triples - self.batch_size > 0:
                batch_idx = np.arange(current_idx, current_idx+self.batch_size)
                current_idx += self.batch_size
                remaining_triples -= self.batch_size
            else:
                batch_idx = np.arange(current_idx, self.store.shape[0])
                remaining_triples = 0
            batch = self.store[batch_idx, :]
            e1 = batch[:,0]
            r = batch[:, 1]
            e2 = batch[:, 2]
            all_e2s = []
            for i in range(e1.shape[0]):
                all_e2s.append(self.store_all_correct[(e1[i], r[i])])
            assert e1.shape[0] == e2.shape[0] == r.shape[0] == len(all_e2s)
            yield e1, r, e2, all_e2s
