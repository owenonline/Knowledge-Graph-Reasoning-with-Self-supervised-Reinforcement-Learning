#!/bin/bash

base_model=$1
dataset=$2
experiment_name=$3
moving_average_window=$4
source "configs/$base_model/$dataset.sh"

path="out/$base_model/$dataset/$experiment_name"

if [ $base_model == "MINERVA" ]
then
    export PYTHONPATH=`pwd`
    echo $PYTHONPATH
    
    python src/process_minerva.py --experiment $path --n $moving_average_window --iterations_rl $total_iterations --iterations_sl $total_iterations_sl
elif [ $base_model == "MultiHopKG-ConvE"]
then
    cd src/ConvE
    export PYTHONPATH=`pwd`
    echo $PYTHONPATH

    python src/process_conve.py --experiment $path --n $moving_average_window --iterations_rl $total_iterations --iterations_sl $total_iterations_sl
else
    echo Invalid RL base model specified; taking no action
fi

echo "experiment $experiment_name processed"