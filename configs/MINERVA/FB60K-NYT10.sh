#!/usr/bin/env bash

base_output_dir="../../out/MINERVA/fb60k-nyt10/"
path_length=3
hidden_size=50
embedding_size=50
batch_size=64
beta=0.2
Lambda=0.02
learning_rate=1e-3
learning_rate_sl=1e-3
use_entity_embeddings=0
train_entity_embeddings=0
train_relation_embeddings=1
data_input_dir="datasets/data_preprocessed/FB60K-NYT10/"
vocab_dir="datasets/data_preprocessed/FB60K-NYT10/vocab"
load_model=0
sl_checkpoints=10
model_name="FB60K-NYT10_reltest"
total_iterations_sl=1000
total_iterations=6000