import numpy as np
import tensorflow as tf


class Agent(tf.keras.Model):

    def __init__(self, params):
        super(Agent, self).__init__()
        self.action_vocab_size = len(params['relation_vocab'])
        self.entity_vocab_size = len(params['entity_vocab'])
        self.embedding_size = params['embedding_size']
        self.hidden_size = params['hidden_size']
        self.ePAD = tf.constant(params['entity_vocab']['PAD'], dtype=tf.int32)
        self.rPAD = tf.constant(params['relation_vocab']['PAD'], dtype=tf.int32)

        self.train_entities = params['train_entity_embeddings']
        self.train_relations = params['train_relation_embeddings']

        self.num_rollouts = params['num_rollouts']
        self.test_rollouts = params['test_rollouts']
        self.LSTM_Layers = params['LSTM_layers']
        self.batch_size = params['batch_size'] * params['num_rollouts']
        self.dummy_start_label = tf.constant(
            np.ones(self.batch_size, dtype='int64') * params['relation_vocab']['DUMMY_START_RELATION'])

        self.entity_embedding_size = self.embedding_size
        self.use_entity_embeddings = params['use_entity_embeddings']
        if self.use_entity_embeddings:
            self.m = 4
        else:
            self.m = 2
        self.dense1 = tf.keras.layers.Dense(4 * self.hidden_size, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(self.m * self.embedding_size, activation=tf.nn.relu)
 
            
        initializer_GloUni = tf.initializers.GlorotUniform()
        if params['use_entity_embeddings']:
            entity_initializer = tf.initializers.GlorotUniform()
        else:
            entity_initializer = tf.constant_initializer(value=0.0)
            
            
        relation_shape=[self.action_vocab_size, 2 * self.embedding_size]
        
        self.relation_lookup_table = tf.Variable(initializer_GloUni(shape=relation_shape),
                                                         shape=relation_shape,
                                                         trainable=self.train_relations)
        if params['pretrained_embeddings_action'] != '':
            action_embedding = np.loadtxt(open(params['pretrained_embeddings_entity'] ))
            self.relation_lookup_table.assign(action_embedding)


        entity_shape= [self.entity_vocab_size, 2 * self.entity_embedding_size]
        
        self.entity_lookup_table = tf.Variable(entity_initializer(shape=entity_shape),
                                                   shape=entity_shape,
                                                   trainable=self.train_entities)
        if params['pretrained_embeddings_entity'] != '':
            entity_embedding = np.loadtxt(open(params['pretrained_embeddings_entity'] ))
            self.entity_lookup_table.assign(entity_embedding)

        cells = []
        for _ in range(self.LSTM_Layers):
            cells.append(tf.keras.layers.LSTMCell(self.m * self.hidden_size))
        self.policy_step = tf.keras.layers.StackedRNNCells(cells)
        
        self.state_init = self.policy_step.get_initial_state(batch_size=self.batch_size, dtype=tf.float32)
        self.relation_init = self.dummy_start_label
    
    def get_query_embedding(self,query_relation):
        query_embedding = tf.nn.embedding_lookup(params=self.relation_lookup_table, ids=query_relation)  # [B, 2D]
        return query_embedding
    
    def get_mem_shape(self):
        return (self.LSTM_Layers, 2, None, self.m * self.hidden_size)

    def policy_MLP(self, state):       
        hidden = self.dense1(state) 
        output = self.dense2(hidden) 
        return output

    def action_encoder(self, next_relations, next_entities):
        with tf.compat.v1.variable_scope("lookup_table_edge_encoder"):
            relation_embedding = tf.nn.embedding_lookup(params=self.relation_lookup_table, ids=next_relations)
            entity_embedding = tf.nn.embedding_lookup(params=self.entity_lookup_table, ids=next_entities)
            if self.use_entity_embeddings:
                action_embedding = tf.concat([relation_embedding, entity_embedding], axis=-1)
            else:
                action_embedding = relation_embedding
        return action_embedding

    def step(self, next_relations, next_entities, prev_state, prev_relation, query_embedding, current_entities,
              range_arr):

        prev_action_embedding = self.action_encoder(prev_relation, current_entities)

        output, new_state = self.policy_step(prev_action_embedding, prev_state)  

        prev_entity = tf.nn.embedding_lookup(params=self.entity_lookup_table, ids=current_entities)
        if self.use_entity_embeddings:
            state = tf.concat([output, prev_entity], axis=-1)
        else:
            state = output

        candidate_action_embeddings = self.action_encoder(next_relations, next_entities)
        state_query_concat = tf.concat([state, query_embedding], axis=-1)

        # MLP for policy

        output = self.policy_MLP(state_query_concat)
        output_expanded = tf.expand_dims(output, axis=1)  
        prelim_scores = tf.reduce_sum(input_tensor=tf.multiply(candidate_action_embeddings, output_expanded), axis=2)

        # Masking PAD actions

        comparison_tensor = tf.ones_like(next_relations, dtype=tf.int32) * self.rPAD  # matrix to compare
        mask = tf.equal(next_relations, comparison_tensor)  # The mask
        dummy_scores = tf.ones_like(prelim_scores) * -99999.0  # the base matrix to choose from if dummy relation
        scores = tf.compat.v1.where(mask, dummy_scores, prelim_scores)  

        action = tf.cast(tf.random.categorical(logits=scores, num_samples=1), dtype=tf.int32)  

        label_action =  tf.squeeze(action, axis=1)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=label_action)  

        action_idx = tf.squeeze(action)
        chosen_relation = tf.gather_nd(next_relations, tf.transpose(a=tf.stack([range_arr, action_idx])))

        return loss, new_state, tf.nn.log_softmax(scores), action_idx, chosen_relation, prelim_scores


