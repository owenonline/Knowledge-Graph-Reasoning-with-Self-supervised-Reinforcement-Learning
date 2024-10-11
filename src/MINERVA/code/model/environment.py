from __future__ import absolute_import
from __future__ import division
import numpy as np
from code.data.feed_data import RelationEntityBatcher
from code.data.grapher import RelationEntityGrapher
from code.data.label_gen import Labeller
import logging
import tensorflow as tf

logger = logging.getLogger()


class Episode(object):

    def __init__(self, graph, labeller, data, params, rl, orig):
        self.rl = rl
        self.orig = orig
        self.grapher = graph
        self.labeller = labeller
        self.batch_size, self.path_len, num_rollouts, test_rollouts, positive_reward, negative_reward, mode, batcher = params
        self.mode = mode

        if self.mode == 'train':
            self.num_rollouts = num_rollouts
        else:
            self.num_rollouts = test_rollouts

        self.current_hop = 0

        if self.rl:
            start_entities, query_relation,  end_entities, all_answers = data
        else:
            start_entities, query_relation,  end_entities, all_answers, self.correct_path = data

        self.no_examples = start_entities.shape[0]
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward
        start_entities = np.repeat(start_entities, self.num_rollouts) 
        batch_query_relation = np.repeat(query_relation, self.num_rollouts)
        end_entities = np.repeat(end_entities, self.num_rollouts)
        self.start_entities = start_entities
        self.end_entities = end_entities
        self.current_entities = np.array(start_entities)
        self.query_relation = batch_query_relation
        self.all_answers = all_answers

        next_actions = self.grapher.return_next_actions(self.current_entities, self.start_entities, self.query_relation,
                                                        self.end_entities, self.all_answers, self.current_hop == self.path_len - 1,
                                                        self.num_rollouts)
        self.state = {}
        self.state['next_relations'] = tf.convert_to_tensor(next_actions[:, :, 1], dtype=tf.int32)
        self.state['next_entities'] = tf.convert_to_tensor(next_actions[:, :, 0], dtype=tf.int32)
        self.state['current_entities'] = tf.convert_to_tensor(self.current_entities, dtype=tf.int32)

    def get_many_to_one(self):
        # check if the number of actions at e1 is greater than the number of actions at e2
        many_to_one = [self.grapher.get_num_actions(e1) > self.grapher.get_num_actions(e2) for (e1, e2) in zip(self.start_entities, self.end_entities)]
        return many_to_one

    def get_state(self):
        return self.state

    def get_query_relation(self):
        return self.query_relation

    def backtrack(self, batch):
        # returns all the actions which, taken at the current state, will take the agent to its previous state
        # this allows the agent to learn to backtrack when it makes a mistake
        return np.where(self.state['next_entities'][batch, :] == self.last_entities[batch])[0]

    def get_reward(self):
        reward = (self.current_entities == self.end_entities)

        # set the True and False values to the values of positive and negative rewards.
        condlist = [reward == True, reward == False]
        choicelist = [self.positive_reward, self.negative_reward]
        reward = np.select(condlist, choicelist)  # [B,]
        return reward

    def __call__(self, action):
        self.current_hop += 1
        self.last_entities=self.current_entities
        batch_indices = tf.range(tf.shape(self.current_entities)[0])
        self.current_entities = tf.gather_nd(
            self.state['next_entities'], 
            tf.stack([batch_indices, action], axis=1)
        )

        next_actions = self.grapher.return_next_actions(self.current_entities, self.start_entities, self.query_relation,
                                                        self.end_entities, self.all_answers, self.current_hop == self.path_len - 1,
                                                        self.num_rollouts )

        self.state['next_relations'] = next_actions[:, :, 1]
        self.state['next_entities'] = next_actions[:, :, 0]
        self.state['current_entities'] = self.current_entities

        return self.state


class env(object):
    def __init__(self, params, mode, rl, orig):
        self.rl = rl
        self.orig = orig
        self.batch_size = params['batch_size']
        self.num_rollouts = params['num_rollouts']
        self.positive_reward = params['positive_reward']
        self.negative_reward = params['negative_reward']
        self.mode = mode
        self.path_len = params['path_length']
        self.test_rollouts = params['test_rollouts']
        if mode == 'train':
            self.batcher = RelationEntityBatcher(rl=self.rl, orig=self.orig,
                                                  dataset=params['dataset'],
                                                  batch_size=params['batch_size'],
                                                  entity_vocab=params['entity_vocab'],
                                                  relation_vocab=params['relation_vocab'])
        else:
            self.batcher = RelationEntityBatcher(rl=self.rl, orig=self.orig,
                                                  dataset=params['dataset'],
                                                  batch_size=params['batch_size'],
                                                  entity_vocab=params['entity_vocab'],
                                                  relation_vocab=params['relation_vocab'],
                                                  mode=mode)

        self.total_no_examples = self.batcher.store.shape[0]
        self.grapher = RelationEntityGrapher(triple_store=params['dataset']['graph'],
                                              max_num_actions=params['max_num_actions'],
                                              entity_vocab=params['entity_vocab'],
                                              relation_vocab=params['relation_vocab'])

        self.labeller = Labeller([self.grapher.array_store, 
                                  params['entity_vocab']['PAD'], 
                                  params['relation_vocab']['PAD'],
                                  self.batcher.store_all_correct])

    def get_episodes(self):
        params = self.batch_size, self.path_len, self.num_rollouts, self.test_rollouts, self.positive_reward, self.negative_reward, self.mode, self.batcher
        if self.mode == 'train':
            for data in self.batcher.yield_next_batch_train(self.labeller):
                yield Episode(self.grapher, self.labeller, data, params, self.rl, self.orig)
        else:
            for data in self.batcher.yield_next_batch_test():
                if data == None:
                    return
                yield Episode(self.grapher, self.labeller, data, params, True, True)
