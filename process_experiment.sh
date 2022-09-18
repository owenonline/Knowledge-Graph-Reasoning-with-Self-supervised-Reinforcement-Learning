#!/bin/bash

base_model=$1
dataset=$2
experiment_name=$3
moving_average_window=$4
source "configs/$base_model/$dataset.sh"

path = "$base_model/$dataset/$experiment_name"

if [ $base_model == "MINERVA" ]
then
    cd src/MINERVA_tf2/datasets/data_preprocessed
    export PYTHONPATH=`pwd`
    echo $PYTHONPATH
    
    python src/process_minerva.py --experiment $experiment_name --n $moving_average_window --iterations_rl $total_iterations --iterations_sl $total_iterations_sl
elif [ $base_model == "MultiHopKG-ConvE"]
then
    cd src/MultiHopKG
    export PYTHONPATH=`pwd`
    echo $PYTHONPATH

    python src/process_conve.py --experiment $experiment_name --n $moving_average_window --iterations_rl $total_iterations --iterations_sl $total_iterations_sl
else
    echo Invalid RL base model specified; taking no action
fi

echo "experiment $experiment_name processed"