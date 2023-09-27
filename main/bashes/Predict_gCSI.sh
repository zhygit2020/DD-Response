#!/usr/bin/bash

python -u ../main.py --source gCSI --task predict --n_epochs 128 --gpu -1 --batch_size 128 --lr 5e-4 --monitor loss_val --metric ACC

# sbatch -p gpu -w gpu09 --gpus=1 -o Predict_ongCSI.log Predict_gCSI.sh
# sbatch -w cu01 -o Predict_ongCSI.log Predict_gCSI.sh