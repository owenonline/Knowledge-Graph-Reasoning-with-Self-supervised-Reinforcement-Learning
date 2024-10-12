from collections import defaultdict
import logging
import numpy as np
import tensorflow as tf
# import csv

logger = logging.getLogger(__name__)


class RelationEntityGrapher:
    def __init__(self, triple_store, relation_vocab, entity_vocab, max_num_actions):

        self.ePAD = entity_vocab['PAD']
        self.rPAD = relation_vocab['PAD']
        self.triple_store = triple_store
        self.relation_vocab = relation_vocab
        self.entity_vocab = entity_vocab
        self.store = defaultdict(list)
        self.array_store = np.ones((len(entity_vocab), max_num_actions, 2), dtype=np.int32)
        self.array_store[:, :, 0] *= self.ePAD
        self.array_store[:, :, 1] *= self.rPAD
        self.masked_array_store = None

        self.rev_relation_vocab = dict([(v, k) for k, v in relation_vocab.items()])
        self.rev_entity_vocab = dict([(v, k) for k, v in entity_vocab.items()])
        self.create_graph()
        print("KG constructed")

    def get_num_actions(self, e):
        return np.where(self.array_store[e,:,0] != self.ePAD)[0].shape[0]

    def create_graph(self):
        # with open(self.triple_store) as triple_file_raw:
            # triple_file = csv.reader(triple_file_raw, delimiter='\t')
        for line in self.triple_store:
            e1 = self.entity_vocab[line[0]]
            r = self.relation_vocab[line[1]]
            e2 = self.entity_vocab[line[2]]
            self.store[e1].append((r, e2))

        for e1 in self.store:
            num_actions = 1
            self.array_store[e1, 0, 1] = self.relation_vocab['NO_OP']
            self.array_store[e1, 0, 0] = e1
            for r, e2 in self.store[e1]:
                if num_actions == self.array_store.shape[1]:
                    break
                self.array_store[e1,num_actions,0] = e2
                self.array_store[e1,num_actions,1] = r
                num_actions += 1
        del self.store
        self.store = None
        self.array_store = tf.convert_to_tensor(self.array_store)

    def return_next_actions(self, current_entities, start_entities, query_relations, answers, all_correct_answers, last_step, rollouts):
        ret = tf.gather(self.array_store, current_entities)
        start_entities = tf.convert_to_tensor(start_entities, dtype=tf.int32)
        query_relations = tf.convert_to_tensor(query_relations, dtype=tf.int32)
        answers = tf.convert_to_tensor(answers, dtype=tf.int32)
        # if np.any(np.isnan(ret[:, :, 0])):
        #     print("ret contains nan")
        for i in range(current_entities.shape[0]):
            if current_entities[i] == start_entities[i]:
                relations = ret[i, :, 1]
                entities = ret[i, :, 0]
                mask = tf.logical_and(
                    tf.equal(ret[:, :, 1], query_relations[:, tf.newaxis]),
                    tf.equal(ret[:, :, 0], answers[:, tf.newaxis])
                )
                ret_entities = tf.where(mask, self.ePAD, ret[:, :, 0])
                ret_relations = tf.where(mask, self.rPAD, ret[:, :, 1])
                ret = tf.stack([ret_entities, ret_relations], axis=-1)
            # if last_step:
            #     entities = ret[i, :, 0]
            #     relations = ret[i, :, 1]

            #     correct_e2 = answers[i]
            #     for j in range(entities.shape[0]):
            #         #print(i/rollouts,j,i,rollouts)
            #         # print(type(entities))
            #         # print(type(all_correct_answers))
            #         # print(entities[i])
            #         # print(all_correct_answers[int(i/rollouts)])
            #         # if entities[j] in all_correct_answers[int(i/rollouts)] and entities[j] != correct_e2:
            #         #     entities[j] = self.ePAD
            #         #     relations[j] = self.rPAD

            #         is_in_all_correct_answers = tf.reduce_any(tf.equal(entities[j], all_correct_answers[int(i/rollouts)]))
            #         is_not_correct_e2 = tf.not_equal(entities[j], correct_e2)
            #         condition = tf.logical_and(is_in_all_correct_answers, is_not_correct_e2)
                    
            #         # Use tf.cond to perform conditional assignment
            #         entities[j] = tf.cond(
            #             condition,
            #             lambda: self.ePAD,
            #             lambda: entities[j]
            #         )
            #         relations[j] = tf.cond(
            #             condition,
            #             lambda: self.rPAD,
            #             lambda: relations[j]
            #         )

        return ret
