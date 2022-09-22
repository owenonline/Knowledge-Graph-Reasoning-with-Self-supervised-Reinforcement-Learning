from __future__ import absolute_import
from __future__ import division
from matplotlib import pyplot as plt
import matplotlib
from tqdm import tqdm
import os
import csv
import logging
import numpy as np
import tensorflow as tf
from code.model.agent import Agent
from code.options import read_options
from code.model.environment import env
import codecs
from collections import defaultdict
import sys
import copy
from code.model.baseline import ReactiveBaseline
from scipy.special import logsumexp as lse
from code.data.vocab_gen import Vocab_Gen
matplotlib.use("agg")
logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

class Trainer(object):
    def __init__(self, params, train_type, reward_type):
        # transfer parameters to self
        for key, val in params.items(): setattr(self, key, val)

        self.rl = train_type == "reinforcement"
        self.original_reward = reward_type == "original"

        self.agent = Agent(params)
        self.save_path = None
        self.train_environment = env(params, 'train', self.rl, self.original_reward)
        self.dev_test_environment = env(params, 'dev', True, True)
        self.test_test_environment = env(params, 'test', True, True)
        self.test_environment = self.dev_test_environment
        self.rev_relation_vocab = self.train_environment.grapher.rev_relation_vocab
        self.rev_entity_vocab = self.train_environment.grapher.rev_entity_vocab
        self.max_hits_at_10 = 0
        self.ePAD = self.entity_vocab['PAD']
        self.rPAD = self.relation_vocab['PAD']
        self.global_step = 0
        self.decaying_beta = tf.keras.optimizers.schedules.ExponentialDecay(self.beta,decay_steps=200,decay_rate=0.90, staircase=True)

        # optimize
        self.baseline = ReactiveBaseline(l=self.Lambda)
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

        # prepare to collect the agent's score on the different relations
        if not self.rl:
            self.relation_counts = defaultdict(int)
            self.relation_scores = defaultdict(int)
            self.total_iterations = self.total_iterations_sl

    def calc_reinforce_loss(self,cum_discounted_reward,loss_all,logits_all):
        loss = tf.stack(loss_all, axis=1)  # [B, T]

        self.tf_baseline = self.baseline.get_baseline_value()

        # multiply with rewards
        final_reward = cum_discounted_reward - self.tf_baseline
        reward_mean, reward_var = tf.nn.moments(x=final_reward, axes=[0, 1])

        # Constant added for numerical stability
        reward_std = tf.sqrt(reward_var) + 1e-6
        final_reward = tf.math.divide(final_reward - reward_mean, reward_std)

        loss = tf.multiply(loss, final_reward)  # [B, T]
        total_loss = tf.reduce_mean(loss) - self.decaying_beta(self.global_step) * self.entropy_reg_loss(logits_all)  # scalar

        return total_loss

    # the scores actually aren't exactly between 0 and 1, so we normalize them to that range to get a proper CCE error
    def normalize_scores(self, scores):
        scores = tf.cast(scores, dtype=tf.float32)
        scores = tf.divide(tf.subtract(scores, tf.reduce_min(scores)), tf.subtract(tf.reduce_max(scores), tf.reduce_min(scores)))

        # if a row in scores is all 0s, change it to a very small nonzero value instead so cce can't produce nans
        if tf.math.reduce_min(tf.math.count_nonzero(scores, axis=1)).numpy() == 0:
            scores = scores + tf.fill(tf.shape(scores), 0.0001)

        return scores

    # returns a new correct vector where 0 values are replaced with whatever the agent already has there
    def positive_cce_masking(self, normalized_scores, label):
        score_cloned = list(tf.identity(normalized_scores).numpy())
        label = list(label)
        
        return np.array([[score if (correct == 0 and correct < self.ignore_threshold) else correct for (score, correct) in zip(batch_score, batch_correct)] for (batch_score, batch_correct) in zip(score_cloned, label)])

    def entropy_reg_loss(self, all_logits):
        all_logits = tf.stack(all_logits, axis=2)
        entropy_policy = - tf.reduce_mean(tf.reduce_sum(tf.multiply(tf.exp(all_logits), all_logits), axis=1))

        return entropy_policy

    def calc_cum_discounted_reward(self, rewards):
        """
        calculates the cumulative discounted reward.
        :param rewards:
        :param T:
        :param gamma:
        :return:
        """
        running_add = np.zeros([rewards.shape[0]])  
        cum_disc_reward = np.zeros([rewards.shape[0], self.path_length])  

        # set the last time step to the reward received at the last state
        cum_disc_reward[:, self.path_length - 1] = rewards
        for t in reversed(range(self.path_length)):
            running_add = self.gamma * running_add + cum_disc_reward[:, t]
            cum_disc_reward[:, t] = running_add

        return cum_disc_reward

    def train(self):
        train_loss = 0.0
        self.batch_counter = 0
        self.first_state_of_test = False
        self.range_arr = np.arange(self.batch_size*self.num_rollouts)

        for episode in self.train_environment.get_episodes():

            if not self.rl:
                for relation in episode.query_relation:
                    self.relation_counts[relation] += 1

            # get initial values
            self.batch_counter += 1
            model_state = self.agent.state_init
            prev_relation = self.agent.relation_init            
            query_relation = episode.get_query_relation()
            query_embedding = self.agent.get_query_embedding(query_relation)
            state = episode.get_state()

            # for use with SL
            last_step = [("N/A",)]*(self.batch_size*self.num_rollouts)

            # for each time step
            with tf.GradientTape() as tape:
                supervised_learning_loss = []
                loss_before_regularization = []
                logits_all = []

                for i in range(self.path_length):
                    loss, model_state, logits, idx, prev_relation, scores = self.agent.step(state['next_relations'],
                                                                                  state['next_entities'],
                                                                                  model_state, prev_relation, query_embedding,
                                                                                  state['current_entities'],  
                                                                                  range_arr=self.range_arr)
                    if self.rl:
                        loss_before_regularization.append(loss)
                        logits_all.append(logits)
                    else:
                        normalized_scores = self.normalize_scores(scores)

                        active_length=scores.shape[0]
                        choices=scores.shape[1]

                        correct=np.full((active_length,choices),0)

                        for batch_num in range(len(episode.correct_path[i])):
                            try:
                                valid = episode.correct_path[i][batch_num][last_step[batch_num]]
                            except:
                                valid = episode.backtrack(batch_num)

                            # if no paths were found, set the label equal to the score so nothing gets changed
                            if len(valid) == 1 and valid[0] == -1:
                                correct[np.array([batch_num]*len(valid), int), :] = normalized_scores[batch_num]
                            else:
                                correct[np.array([batch_num]*len(valid), int),np.array(valid, int)] = np.ones(len(valid))

                        current_actions = idx.numpy()
                        last_step = [tuple(list(x) + [y]) for (x, y) in zip(last_step, current_actions)]

                        tensorized = tf.convert_to_tensor(correct)
                        normalized = self.normalize_scores(scores)
                        loss = self.cce(tensorized, normalized)
                        
                        supervised_learning_loss.append(loss)

                    state = episode(idx)

                # get the final reward from the environment
                rewards = episode.get_reward()

                if not self.rl:

                    # update the list of incorrect queries. We have a dictionary with the number of times the agent got the query wrong
                    # and a list of the actual queries sorted by the values stored in that dictionary. That way, the queries the agent
                    # has gotten wrong the most stay at the top
                    correct_queries = [[e1, r, e2] for (e1, r, e2, reward) in zip(episode.start_entities, episode.query_relation, episode.end_entities, rewards) if reward == episode.positive_reward]
                    for query in correct_queries:
                        self.relation_scores[query[1]] += 1

                # compute loss
                if self.rl:
                    cum_discounted_reward = self.calc_cum_discounted_reward(rewards)  # [B, T]
                    batch_total_loss = self.calc_reinforce_loss(cum_discounted_reward,loss_before_regularization,logits_all)
                else:
                    sl_loss_float64 = [tf.cast(x, tf.float64) for x in supervised_learning_loss]
                    reduced_sum = tf.reduce_sum(sl_loss_float64,0)
                    square = tf.math.square(reduced_sum)
                    supervised_learning_total_loss = tf.math.reduce_mean(square)

            # update weights
            if self.rl:
                gradients = tape.gradient(batch_total_loss, self.agent.trainable_variables)
            else:
                gradients = tape.gradient(supervised_learning_total_loss, self.agent.trainable_variables)

            gradients, _ = tf.clip_by_global_norm(gradients, self.grad_clip_norm)
            self.optimizer.apply_gradients(zip(gradients, self.agent.trainable_variables))        

            # print statistics
            train_loss = 0.98 * train_loss + 0.02 * (batch_total_loss if self.rl else supervised_learning_total_loss)
            avg_reward = np.mean(rewards)
            reward_reshape = np.reshape(rewards, (self.batch_size, self.num_rollouts))
            reward_reshape = np.sum(reward_reshape, axis=1)
            reward_reshape = (reward_reshape > 0)
            num_ep_correct = np.sum(reward_reshape)
            if np.isnan(train_loss):
                raise ArithmeticError("Error in computing loss")
                
            logger.info("RL: {0:4d}, batch_counter: {1:4d}, num_hits: {2:7.4f}, avg. reward per batch {3:7.4f}, "
                        "num_ep_correct {4:4d}, avg_ep_correct {5:7.4f}, train loss {6:7.4f}".
                        format(int(self.rl),self.batch_counter, np.sum(rewards), avg_reward, num_ep_correct,
                                (num_ep_correct / self.batch_size),
                                train_loss))                

            if self.rl and self.batch_counter%self.eval_every == 0:
                with open(self.output_dir + '/scores.txt', 'a') as score_file:
                    score_file.write("In-progress score batch " + str(self.batch_counter))
                self.test(beam=True, extras=False)

            if self.batch_counter >= self.total_iterations:
                if self.rl:
                    return
                else:
                    viewed_labels = self.train_environment.labeller.viewed_labels
                    self.train_environment.labeller.viewed_labels = 0
                    return viewed_labels

    def test(self, beam=False, extras=True):
        batch_counter = 0

        all_final_reward_1 = 0
        all_final_reward_3 = 0
        all_final_reward_5 = 0
        all_final_reward_10 = 0
        all_final_reward_20 = 0
        auc = 0

        if extras:
            all_final_reward_1_to_one = 0
            all_final_reward_3_to_one = 0
            all_final_reward_10_to_one = 0

            all_final_reward_1_to_many = 0
            all_final_reward_3_to_many = 0
            all_final_reward_10_to_many = 0

            total_examples = self.test_environment.total_no_examples
            total_examples_to_one = 0
            total_examples_to_many = 0
        for episode in tqdm(self.test_environment.get_episodes()):
            batch_counter += 1

            if extras:
                many_to_one = episode.get_many_to_one()

            temp_batch_size = episode.no_examples
            
            query_relation = episode.get_query_relation()
            query_embedding = self.agent.get_query_embedding(query_relation)
            
            # set initial beam probs
            beam_probs = np.zeros((temp_batch_size * self.test_rollouts, 1))

            # get initial state
            state = episode.get_state()
            
            mem = self.agent.get_mem_shape()
            agent_mem = np.zeros((mem[0], mem[1], temp_batch_size*self.test_rollouts, mem[3]) ).astype('float32')
            layer_state = tf.unstack(agent_mem, self.LSTM_layers)
            model_state = [tf.unstack(s, 2) for s in layer_state]            
            previous_relation = np.ones((temp_batch_size * self.test_rollouts, ), dtype='int64') * self.relation_vocab[
                'DUMMY_START_RELATION']
            self.range_arr_test = np.arange(temp_batch_size * self.test_rollouts)

            self.log_probs = np.zeros((temp_batch_size*self.test_rollouts,)) * 1.0

            for i in range(self.path_length):
                if i == 0:
                    self.first_state_of_test = True

                loss, agent_mem, test_scores, test_action_idx, chosen_relation, _ = self.agent.step(state['next_relations'],
                                                                              state['next_entities'],
                                                                              model_state, previous_relation, query_embedding,
                                                                              state['current_entities'],  
                                                                              range_arr=self.range_arr_test)
                agent_mem = tf.stack(agent_mem)
                agent_mem = agent_mem.numpy()
                test_scores = test_scores.numpy()
                test_action_idx = test_action_idx.numpy()
                chosen_relation = chosen_relation.numpy()
                if beam:
                    k = self.test_rollouts
                    new_scores = test_scores + beam_probs
                    if i == 0:
                        idx = np.argsort(new_scores)
                        idx = idx[:, -k:]
                        ranged_idx = np.tile([b for b in range(k)], temp_batch_size)
                        idx = idx[np.arange(k*temp_batch_size), ranged_idx]
                    else:
                        idx = self.top_k(new_scores, k)

                    y = idx//self.max_num_actions
                    x = idx%self.max_num_actions

                    y += np.repeat([b*k for b in range(temp_batch_size)], k)
                    state['current_entities'] = state['current_entities'][y]
                    state['next_relations'] = state['next_relations'][y,:]
                    state['next_entities'] = state['next_entities'][y, :]

                    agent_mem = agent_mem[:, :, y, :]
                    test_action_idx = x
                    chosen_relation = state['next_relations'][np.arange(temp_batch_size*k), x]
                    beam_probs = new_scores[y, x]
                    beam_probs = beam_probs.reshape((-1, 1))
                previous_relation = chosen_relation
                layer_state = tf.unstack(agent_mem, self.LSTM_layers)
                model_state = [tf.unstack(s, 2) for s in layer_state]   
                state = episode(test_action_idx)
                self.log_probs += test_scores[np.arange(self.log_probs.shape[0]), test_action_idx]
            if beam:
                self.log_probs = beam_probs

            # ask environment for final reward
            rewards = episode.get_reward()
            reward_reshape = np.reshape(rewards, (temp_batch_size, self.test_rollouts))
            self.log_probs = np.reshape(self.log_probs, (temp_batch_size, self.test_rollouts))
            sorted_indx = np.argsort(-self.log_probs)
            final_reward_1 = 0
            final_reward_3 = 0
            final_reward_5 = 0
            final_reward_10 = 0
            final_reward_20 = 0

            if extras:
                final_reward_1_to_one = 0
                final_reward_3_to_one = 0
                final_reward_10_to_one = 0

                final_reward_1_to_many = 0
                final_reward_3_to_many = 0
                final_reward_10_to_many = 0

            AP = 0
            ce = episode.state['current_entities'].reshape((temp_batch_size, self.test_rollouts))
            se = episode.start_entities.reshape((temp_batch_size, self.test_rollouts))

            # we only look at the first rollout since all of them will have the same number as these are based on the query
            if extras:
                many_to_one = np.array(many_to_one).reshape((temp_batch_size, self.test_rollouts))[:, 0]
            for b in range(temp_batch_size):
                answer_pos = None
                seen = set()
                pos=0
                if self.pool == 'max':
                    for r in sorted_indx[b]:
                        if reward_reshape[b,r] == self.positive_reward:
                            answer_pos = pos
                            break
                        if ce[b, r] not in seen:
                            seen.add(ce[b, r])
                            pos += 1
                if self.pool == 'sum':
                    scores = defaultdict(list)
                    answer = ''
                    for r in sorted_indx[b]:
                        scores[ce[b,r]].append(self.log_probs[b,r])
                        if reward_reshape[b,r] == self.positive_reward:
                            answer = ce[b,r]
                    final_scores = defaultdict(float)
                    for e in scores:
                        final_scores[e] = lse(scores[e])
                    sorted_answers = sorted(final_scores, key=final_scores.get, reverse=True)
                    if answer in  sorted_answers:
                        answer_pos = sorted_answers.index(answer)
                    else:
                        answer_pos = None

                # add answer pos for all
                if answer_pos != None:
                    if answer_pos < 20:
                        final_reward_20 += 1
                        if answer_pos < 10:
                            if extras:
                                if many_to_one[b]:
                                    final_reward_10_to_many += 1
                                    total_examples_to_many += 1
                                else:
                                    final_reward_10_to_one += 1
                                    total_examples_to_one += 1

                            final_reward_10 += 1
                            if answer_pos < 5:
                                final_reward_5 += 1
                                if answer_pos < 3:
                                    if extras:
                                        if many_to_one[b]:
                                            final_reward_3_to_many += 1
                                            total_examples_to_many += 1
                                        else:
                                            final_reward_3_to_one += 1
                                            total_examples_to_one += 1

                                    final_reward_3 += 1
                                    if answer_pos < 1:
                                        if extras:
                                            if many_to_one[b]:
                                                final_reward_1_to_many += 1
                                                total_examples_to_many += 1
                                            else:
                                                final_reward_1_to_one += 1
                                                total_examples_to_one += 1

                                        final_reward_1 += 1
                if answer_pos == None:
                    AP += 0
                else:
                    AP += 1.0/((answer_pos+1))

            all_final_reward_1 += final_reward_1
            all_final_reward_3 += final_reward_3
            all_final_reward_5 += final_reward_5
            all_final_reward_10 += final_reward_10
            all_final_reward_20 += final_reward_20
            auc += AP

            if extras:
                all_final_reward_1_to_one += final_reward_1_to_one
                all_final_reward_3_to_one += final_reward_3_to_one
                all_final_reward_10_to_one += final_reward_10_to_one

                all_final_reward_1_to_many += final_reward_1_to_many
                all_final_reward_3_to_many += final_reward_3_to_many
                all_final_reward_10_to_many += final_reward_10_to_many

        # tally up total examples for the other categories too
        all_final_reward_1 /= total_examples
        all_final_reward_3 /= total_examples
        all_final_reward_5 /= total_examples
        all_final_reward_10 /= total_examples
        all_final_reward_20 /= total_examples
        auc /= total_examples

        if extras:
            if total_examples_to_one > 0:
                all_final_reward_1_to_one /= total_examples_to_one
                all_final_reward_3_to_one /= total_examples_to_one
                all_final_reward_10_to_one /= total_examples_to_one
            else:
                all_final_reward_1_to_one = -1
                all_final_reward_3_to_one = -1
                all_final_reward_10_to_one = -1


            if total_examples_to_many > 0:
                all_final_reward_1_to_many /= total_examples_to_many
                all_final_reward_3_to_many /= total_examples_to_many
                all_final_reward_10_to_many /= total_examples_to_many
            else:
                all_final_reward_1_to_many = -1
                all_final_reward_3_to_many = -1
                all_final_reward_10_to_many = -1

        with open(self.output_dir + '/scores.txt', 'a') as score_file:
            score_file.write("Hits@1: {0:7.4f}".format(all_final_reward_1))
            score_file.write("\n")
            score_file.write("Hits@3: {0:7.4f}".format(all_final_reward_3))
            score_file.write("\n")
            score_file.write("Hits@5: {0:7.4f}".format(all_final_reward_5))
            score_file.write("\n")
            score_file.write("Hits@10: {0:7.4f}".format(all_final_reward_10))
            score_file.write("\n")
            score_file.write("Hits@20: {0:7.4f}".format(all_final_reward_20))
            score_file.write("\n")
            score_file.write("auc: {0:7.4f}".format(auc))
            score_file.write("\n")
            score_file.write("\n")

            if extras:
                score_file.write("to many correct paths")
                score_file.write("\n")
                score_file.write("Hits@1: {0:7.4f}".format(all_final_reward_1_to_many))
                score_file.write("\n")
                score_file.write("Hits@3: {0:7.4f}".format(all_final_reward_3_to_many))
                score_file.write("\n")
                score_file.write("Hits@10: {0:7.4f}".format(all_final_reward_10_to_many))
                score_file.write("\n")
                score_file.write("\n")

                score_file.write("to one correct paths")
                score_file.write("\n")
                score_file.write("Hits@1: {0:7.4f}".format(all_final_reward_1_to_one))
                score_file.write("\n")
                score_file.write("Hits@3: {0:7.4f}".format(all_final_reward_3_to_one))
                score_file.write("\n")
                score_file.write("Hits@10: {0:7.4f}".format(all_final_reward_10_to_one))
                score_file.write("\n")
                score_file.write("\n")
        
    def top_k(self, scores, k):
        scores = scores.reshape(-1, k * self.max_num_actions)
        idx = np.argsort(scores, axis=1)
        idx = idx[:, -k:]
        return idx.reshape((-1))

if __name__ == '__main__':
    # read command line options
    options = read_options()
    # Set logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    logfile = logging.FileHandler(options['log_file_name'], 'w')
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)
    #read dataset
        
    options['dataset']={}
    Dataset_list=['train','test','dev','graph']
    for dataset in Dataset_list:
        print(os.getcwd())
        input_file = options['data_input_dir']+dataset+'.txt'
        ds = []
        with open(input_file) as raw_input_file:
            csv_file = csv.reader(raw_input_file, delimiter = '\t' )
            for line in csv_file:
                ds.append(line)   
        options['dataset'][dataset]=ds
 
    ds = []
    input_file = options['data_input_dir']+'full_graph.txt'
    if os.path.isfile(input_file):
        with open(input_file) as raw_input_file:
            csv_file = csv.reader(raw_input_file, delimiter = '\t' )
            for line in csv_file:
                ds.append(line)  
    else:
        for dataset in Dataset_list:
            ds = ds + options['dataset'][dataset]
    options['dataset']['full_graph'] = ds       
    
    # read the vocab files, it will be used by many classes hence global scope
    # logger.info('reading vocab files...')
    vocab=Vocab_Gen(Datasets=[options['dataset']['train'],options['dataset']['test'],options['dataset']['graph']])
    options['relation_vocab'] = vocab.relation_vocab
    
    options['entity_vocab'] = vocab.entity_vocab

    print(len(options['entity_vocab'] ))
    # logger.info('Reading mid to name map')
    mid_to_word = {}

    # logger.info('Done..')
    logger.info('Total number of entities {}'.format(len(options['entity_vocab'])))
    logger.info('Total number of relations {}'.format(len(options['relation_vocab'])))
    save_path = ''

    def make_sl_checkpoint(last_epoch, options):
        original_model_dir = options['model_dir']
        
        # make checkpoint folder
        options['output_dir'] += '/checkpoint_sl_epoch_'+str(last_epoch)
        os.mkdir(options['output_dir'])

        # make model folder
        os.mkdir(options['output_dir']+'/model_weights/')
        options['model_dir'] = options['output_dir']+'/model_weights/'

        # make trainer
        trainer = Trainer(options, "reinforcement", "original")
        trainer.agent.load_weights(original_model_dir)

        # do RL training
        trainer.train()

        # do testing
        with open(options['output_dir'] + '/scores.txt', 'a') as score_file:
            score_file.write("Final score: ")
        trainer.test(beam=True)

        # save model
        trainer.agent.save_weights(options['model_dir'] + options['model_name'])

    original_options = copy.deepcopy(options)
    
    # create SL Trainer
    options['learning_rate'] = options['learning_rate_sl']
    sl_trainer = Trainer(options, "supervised", "our")

    # Create checkpoint for pure RL run
    last_epoch = 0
    sl_trainer.agent.save_weights(options['model_dir'])
    make_sl_checkpoint(last_epoch, copy.deepcopy(original_options))
    sl_trainer.agent.load_weights(options['model_dir'])
    
    # Create SL checkpoints
    for ckpt in range(1, options['sl_checkpoints']):
        # train SL, then log the number of new labels seen by the agent and the agent's score on different relations
        viewed_labels = sl_trainer.train()
        logger.info("Ending SL round {}, {} new labels viewed".format(ckpt, viewed_labels))
        for relation in sl_trainer.relation_counts:
            logger.info("relation ckpt {} -- {}: {} correct out of {} appearances; {:.2f}% ({:.2f}% of dataset)".format(ckpt, relation, sl_trainer.relation_scores[relation], sl_trainer.relation_counts[relation], (sl_trainer.relation_scores[relation]/sl_trainer.relation_counts[relation])*100, (sl_trainer.train_environment.batcher.relation_counts[relation]/sl_trainer.train_environment.total_no_examples)*100))

        # train RL and create checkpoint
        sl_trainer.agent.save_weights(options['model_dir'])
        make_sl_checkpoint(ckpt, copy.deepcopy(original_options))
        sl_trainer.agent.load_weights(options['model_dir'])

        

    