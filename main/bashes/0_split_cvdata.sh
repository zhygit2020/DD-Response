#!/usr/bin/bash

python -u ../0_split_cvdata.py --split_data True --sampling NONE

# sbatch -w cu20 0_split_cvdata.sh