#!/bin/bash

base_model=$1
dataset=$2
gpu=$3

if [ $base_model == "MINERVA" ]
then
    cd src/MINERVA_tf2/datasets/data_preprocessed
    export PYTHONPATH=`pwd`
    echo $PYTHONPATH
    
    if [ ! -f "fb60k.tgz"]
    then
        cat fb60k.tgz.parta* > fb60k.tgz
        rm -rf fb60k.tgz.parta*

    [ $dataset == "FB15K-237" ] && [ ! -d "/FB15K-237" ] && tar -C /FB15K-237 -xvzf fb15k.tgz || echo "experiment $base_model-$dataset already prepared"
    [ $dataset == "FB60K-NYT10" ] && [ ! -d "/FB60K-NYT10" ] && tar -C /FB60K-NYT10 -xvzf fb60k.tgz || echo "experiment $base_model-$dataset already prepared"
    [ $dataset == "NELL-995" ] && [ ! -d "/nell-995" ] && tar -C /nell-995 -xvzf nell995.tgz || echo "experiment $base_model-$dataset already prepared"
    [ $dataset == "WN18RR" ] && [ ! -d "/WN18RR" ] && tar -C /WN18RR -xvzf wn18rr.tgz || echo "experiment $base_model-$dataset already prepared"
elif [ $base_model == "MultiHopKG-ConvE"]
then
    NELLFLAG=""
    if [ $dataset == "NELL-995" ]
    then
        NELLFLAG="--test"

    cd src/MultiHopKG
    export PYTHONPATH=`pwd`
    echo $PYTHONPATH

    [ ! -d "/data" ] && tar xvzf data-release.tgz
    echo "Processing $dataset"
    ./experiment.sh configs/$dataset.sh --process_data $gpu $NELLFLAG
    echo "Preparing ConvE model for $dataset"
    ./experiment-emb.sh configs/$dataset-conve.sh --train $gpu $NELLFLAG
else
    echo Invalid RL base model specified; taking no action
fi

echo "experiment $base_model-$dataset prepared; follow instructions in README to run"