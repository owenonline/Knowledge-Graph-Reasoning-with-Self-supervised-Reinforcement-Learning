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
            mkdir "./FB15K-237"
            tar -C ./FB15K-237 -xvzf fb15k.tgz
        else
            echo "experiment $base_model-$dataset already prepared"
            exit 1
        fi
    elif [ $dataset == "FB60K-NYT10" ]
    then
        if [ ! -d "./FB60K-NYT10" ]
        then
            mkdir "./FB60K-NYT10"
            tar -C ./FB60K-NYT10 -xvzf fb60k.tgz
        else
            echo "experiment $base_model-$dataset already prepared"
            exit 1
        fi
    elif [ $dataset == "NELL-995" ]
    then
        if [ ! -d "./nell-995" ]
        then
            mkdir "./nell-995"
            tar -C ./nell-995 -xvzf nell995.tgz
        else
            echo "experiment $base_model-$dataset already prepared"
            exit 1
        fi
    elif [ $dataset == "WN18RR" ]
    then
        if [ ! -d "./WN18RR" ]
        then
            mkdir "./WN18RR"
            tar -C ./WN18RR -xvzf wn18rr.tgz
        else
            echo "experiment $base_model-$dataset already prepared"
            exit 1
        fi
    fi
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

    if [ ! -d "./data" ]
    then
        tar -xvzf data-release.tgz
    fi

    echo "Processing $dataset"
    ./experiment.sh ../../configs/ConvE/$dataset.sh --process_data $gpu $NELLFLAG
    echo "Preparing ConvE model for $dataset"
    ./experiment-emb.sh ../../configs/ConvE/$dataset-conve.sh --train $gpu $NELLFLAG
else
    echo Invalid RL base model specified; taking no action
fi

echo "experiment $base_model-$dataset prepared; follow instructions in README to run"