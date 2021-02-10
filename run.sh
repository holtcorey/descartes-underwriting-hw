#!/bin/bash

SEED=4

python3 prepare_data.py
python3 split_data.py --random_state $SEED

python3 train.py --model XGB --random_state $SEED

python3 evaluate_on_future_validation.py

python3 generate_sub.py