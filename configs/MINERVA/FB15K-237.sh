#!/usr/bin/env bash

base_output_dir="../../out/MINERVA/fb15k-237/"
path_length=3
hidden_size=50
embedding_size=50
batch_size=256
beta=0.02
Lambda=0.02
learning_rate=1e-3
learning_rate_sl=1e-3
use_entity_embeddings=0
train_entity_embeddings=0
train_relation_embeddings=1
data_input_dir="datasets/data_preprocessed/FB15K-237/"
vocab_dir="datasets/data_preprocessed/FB15K-237/vocab"
load_model=0
sl_checkpoints=16
model_name="FB15K-237_pathnum"
total_iterations=7001
total_iterations_sl=1000