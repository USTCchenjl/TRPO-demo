#!/bin/bash
CUDA_VISIBLE_DEVICES=3 \
PYTHONPATH=$PYTHONPATH:~/Project/Learn2Run \
python ../scripts/train.py \
--t_max 400 \
--max_global_steps 1000000 \
--model_dir ../weights/ \