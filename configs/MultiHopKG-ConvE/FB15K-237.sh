#!/usr/bin/env bash

data_dir="data/FB15K-237"
model="point.rs.conve"
group_examples_by_query="False"
use_action_space_bucketing="True"

bandwidth=400
entity_dim=200
relation_dim=200
history_dim=200
history_num_layers=3
num_rollouts=20
num_rollout_steps=3
bucket_interval=10
num_epochs=40
num_wait_epochs=100
num_peek_epochs=2
batch_size=128
train_batch_size=128
dev_batch_size=1
learning_rate=0.001
baseline="n/a"
grad_norm=0
emb_dropout_rate=0.3
ff_dropout_rate=0.1
action_dropout_rate=0.5
action_dropout_anneal_interval=1000
reward_shaping_threshold=0
beta=0.02
relation_only="False"
beam_size=128
sl_checkpoints=6
total_iterations=14875
total_iterations_sl=1000
eval_every=2975

conve_state_dict_path="../../out/ConvE/FB15K-237_conve_embedding/model_best.tar"

num_paths_per_entity=-1
margin=-1