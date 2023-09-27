#!/usr/bin/bash

python -u ../main.py --kfold_num 5 --task cv --n_epochs 128 --gpu 0 --batch_size 128 --lr 5e-4 --monitor loss_val --metric ACC --source GDSC,CCLE,CTRP

# sbatch -p gpu -w gpu10 --gpus=1 -o monitor_loss_bt128.log DRS_molossbt128.sh