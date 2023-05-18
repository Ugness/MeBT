#!/bin/bash

python train_transformer.py --base $1 --gpus $3 --default_root_dir $2 --check_val_every_n_epoch=100 --max_steps 2000000 --accumulate_grad_batches 1
