#!/usr/bin/bash

python -u ../0_map_transfer.py --datatype drug_fea --fitmethod umap --metric cosine --split_data False --scale_method None 

# sbatch -w cu20 0_trans_drug.sh