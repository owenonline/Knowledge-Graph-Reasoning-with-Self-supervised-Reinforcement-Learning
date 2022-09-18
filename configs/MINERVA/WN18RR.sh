#!/usr/bin/env bash

base_output_dir="../../out/MINERVA/WN18RR/"
path_length=3
hidden_size=50
embedding_size=50
batch_size=256
beta=0.05
Lambda=0.05
learning_rate_sl=0.001
use_entity_embeddings=0
train_entity_embeddings=0
train_relation_embeddings=1
data_input_dir="datasets/data_preprocessed/WN18RR/"
vocab_dir="datasets/data_preprocessed/WN18RR/vocab"
load_model=0
sl_checkpoints=10
model_name="WN18RR_reltest"
total_iterations=1000
total_iterations_sl=575
num_cycles=10
