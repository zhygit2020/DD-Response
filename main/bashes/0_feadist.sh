#!/usr/bin/bash

python -u ../0_feadist_fit.py --scale_method standard

# sbatch -w cu20 0_feadist.sh