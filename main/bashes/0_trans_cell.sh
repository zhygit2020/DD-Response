#!/usr/bin/bash

python -u ../0_map_transfer.py --datatype cellline_fea --fitmethod umap --metric cosine --split_data False --scale_method standard
# cosine correlation jaccard 
# umap tsne mds

# sbatch -w cu20 0_trans_cell.sh