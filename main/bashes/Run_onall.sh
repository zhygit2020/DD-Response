#!/usr/bin/bash

python -u ../main.py --source GDSC,CCLE,CTRP --task run --n_epochs 128 --gpu 0 --batch_size 64 --lr 5e-4 --monitor loss_val --metric ACC

# sbatch -p gpu -w gpu09 --gpus=1 -o Run_onall.log Run_onall.sh