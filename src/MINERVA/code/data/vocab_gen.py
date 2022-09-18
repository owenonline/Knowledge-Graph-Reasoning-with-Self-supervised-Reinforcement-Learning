#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 00:07:01 2020

@author: yingma
"""


class Vocab_Gen:
    def __init__(self, Datasets):
        self.entity_vocab = {}
        self.relation_vocab = {}
        
        
        self.entity_vocab['PAD'] = len(self.entity_vocab)
        self.entity_vocab['UNK'] = len(self.entity_vocab)
        self.relation_vocab['PAD'] = len(self.relation_vocab)
        self.relation_vocab['DUMMY_START_RELATION'] = len(self.relation_vocab)
        self.relation_vocab['NO_OP'] = len(self.relation_vocab)
        self.relation_vocab['UNK'] = len(self.relation_vocab)
        
        self.entity_counter = len(self.entity_vocab)
        self.relation_counter = len(self.relation_vocab)
        for dataset in Datasets:
            for line in dataset:
                if line[0] not in self.entity_vocab:
                    self.entity_vocab[line[0]] = self.entity_counter
                    self.entity_counter += 1
                if line[2] not in self.entity_vocab:
                    self.entity_vocab[line[2]] = self.entity_counter
                    self.entity_counter += 1
                if line[1] not in self.relation_vocab:
                    self.relation_vocab[line[1]] = self.relation_counter
                    self.relation_counter += 1 
