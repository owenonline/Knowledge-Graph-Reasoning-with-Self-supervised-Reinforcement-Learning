#!/usr/bin/env bash

base_output_dir="../../out/MINERVA/nell/worksfor"
path_length=3
hidden_size=50
embedding_size=50
batch_size=128
beta=0.05
Lambda=0.02
learning_rate=1e-3
learning_rate_sl=1e-3
use_entity_embeddings=0
train_entity_embeddings=0
train_relation_embeddings=1
data_input_dir="datasets/data_preprocessed/nell/"
vocab_dir="datasets/data_preprocessed/nell/vocab"
load_model=0
sl_checkpoints=10
model_name="NELL995_reltest"
total_iterations=3000
total_iterations_sl=1000