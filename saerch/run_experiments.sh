#!/bin/bash

# Name of your Python script
PYTHON_SCRIPT="topk_sae.py"

date

# Function to run the Python script with given parameters
run_experiment() {
    local k=$1
    local n_dirs=$2
    local csLG=$3
    
    echo "Running experiment: k=$k, n_dirs=$n_dirs, csLG=$csLG"
    
    if [ "$csLG" = true ]; then
        python "$PYTHON_SCRIPT" --k "$k" --n_dirs "$n_dirs" --csLG
    else
        python "$PYTHON_SCRIPT" --k "$k" --n_dirs "$n_dirs"
    fi
    
    echo "Experiment completed."
    echo "-----------------------------------"
}

# Run experiments
# run_experiment 128 9216 false
run_experiment 128 12288 false
# run_experiment 128 3072 true
# run_experiment 128 4608 true
# run_experiment 128 6144 true
# run_experiment 128 9216 true
# run_experiment 128 12288 true

echo "All experiments completed."
date