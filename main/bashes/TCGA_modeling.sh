#!/usr/bin/bash

python -u ../tcga_main.py --kfold_num 5 --n_epochs 128 --gpu 0 --batch_size 128 --lr 5e-4 --monitor loss_val --metric ACC --source GDSC,CCLE,CTRP

# sbatch -p gpu -w gpu10 --gpus=1 -o TCGA_modeling_5e4.log TCGA_modeling.sh
