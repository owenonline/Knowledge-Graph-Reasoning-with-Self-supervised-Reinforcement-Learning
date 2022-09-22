#!/bin/bash

base_model=$1
dataset=$2
gpu=$3
experiment_name=$4
config="../../configs/$base_model/$dataset.sh"

if [ $base_model == "MINERVA" ]
then
    TF_GPU_ALLOCATOR=cuda_malloc_async

    cd src/MINERVA
    export PYTHONPATH=`pwd`
    echo $PYTHONPATH

    gpu_id=$gpu
    source $config
    cmd="python -u -m code.model.trainer --base_output_dir $base_output_dir --path_length $path_length --hidden_size $hidden_size --embedding_size $embedding_size \
        --batch_size $batch_size --beta $beta --Lambda $Lambda --learning_rate $learning_rate --learning_rate_sl $learning_rate_sl \
        --use_entity_embeddings $use_entity_embeddings  --train_entity_embeddings $train_entity_embeddings --train_relation_embeddings $train_relation_embeddings \
        --data_input_dir $data_input_dir --vocab_dir $vocab_dir --load_model $load_model --total_iterations $total_iterations \
        --total_iterations_sl $total_iterations_sl --model_name $experiment_name"

    echo "Executing $cmd"

    CUDA_VISIBLE_DEVICES=$gpu_id $cmd
    cd ../..
elif [ $base_model == "MultiHopKG-ConvE" ]
then
    NELLFLAG=""
    if [ $dataset == "NELL-995" ]
    then
        NELLFLAG="--test"
    fi

    cd src/ConvE
    export PYTHONPATH=`pwd`
    echo $PYTHONPATH

    ./experiment-rs.sh $config --train $gpu --experiment_name $experiment_name $NELLFLAG
    cd ../..
else
    echo "Invalid RL base model specified; taking no action"
fi