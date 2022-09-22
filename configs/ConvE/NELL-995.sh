#!/usr/bin/env bash

data_dir="data/NELL-995"
model="point.rs.conve"
group_examples_by_query="False"
use_action_space_bucketing="True"

bandwidth=256
entity_dim=200
relation_dim=200
history_dim=200
history_num_layers=3
num_rollouts=20
num_rollout_steps=3
num_epochs=30
num_wait_epochs=300
num_peek_epochs=2
bucket_interval=10
batch_size=128
train_batch_size=128
dev_batch_size=1
learning_rate=0.003
baseline="n/a"
grad_norm=5
emb_dropout_rate=0.1
ff_dropout_rate=0.1
action_dropout_rate=0.1
action_dropout_anneal_interval=1000
reward_shaping_threshold=0
beta=0.05
relation_only="False"
beam_size=512
sl_checkpoints=3 #14
total_iterations=10 #2155
total_iterations_sl=7 #293
eval_every=5 #431

conve_state_dict_path="../../out/ConvE/NELL-995.TEST/NELL-995.TEST_conve_embedding/model_best.tar"

num_paths_per_entity=-1
margin=-1