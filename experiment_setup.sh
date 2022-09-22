#!/bin/bash

base_model=$1
dataset=$2
gpu=$3

if [ $base_model == "MINERVA" ]
then
    cd src/MINERVA/datasets/data_preprocessed
    export PYTHONPATH=`pwd`
    echo $PYTHONPATH
    
    if [ ! -f "fb60k.tgz" ]
    then
        cat fb60k.tgz.parta* > fb60k.tgz
        rm -rf fb60k.tgz.parta*
    fi

    if [ $dataset == "FB15K-237" ]
    then
        if [ ! -d "./FB15K-237" ]
        then
            tar -xvzf fb15k.tgz
        else
            echo "experiment $base_model-$dataset already prepared"
            exit 1
        fi
    elif [ $dataset == "FB60K-NYT10" ]
    then
        if [ ! -d "./FB60K-NYT10" ]
        then
            tar -xvzf fb60k.tgz
        else
            echo "experiment $base_model-$dataset already prepared"
            exit 1
        fi
    elif [ $dataset == "NELL-995" ]
    then
        if [ ! -d "./NELL-995" ]
        then
            tar -xvzf nell995.tgz
        else
            echo "experiment $base_model-$dataset already prepared"
            exit 1
        fi
    elif [ $dataset == "WN18RR" ]
    then
        if [ ! -d "./WN18RR" ]
        then
            tar -xvzf wn18rr.tgz
        else
            echo "experiment $base_model-$dataset already prepared"
            exit 1
        fi
    fi
elif [ $base_model == "ConvE" ]
then
    cd src/ConvE
    export PYTHONPATH=`pwd`
    echo $PYTHONPATH

    if [ ! -d "./data" ]
    then
        tar -xvzf data-release.tgz
    fi

    echo "Processing $dataset"
    ./experiment.sh ../../configs/ConvE/$dataset.sh --process_data $gpu
    echo "Preparing ConvE model for $dataset"
    ./experiment-emb.sh ../../configs/ConvE/$dataset-conve.sh --train $gpu
else
    echo "Invalid RL base model specified; taking no action"
fi

echo "experiment $base_model-$dataset prepared; follow instructions in README to run"