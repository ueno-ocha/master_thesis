#!/bin/bash
echo "Running experiment..."
PYTHON=python3

$PYTHON main.py \
    --train_data_file 'result_mlm_new.json' \
    --device 'cuda:3' \
    --learning_rate '5e-6' \
    --num_train_epochs '2' \
    --output_dir 'results' \
    --experiment_config 'column_guided/exp6.yaml' \
    --seed '51'
