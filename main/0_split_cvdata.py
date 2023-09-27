from rdkit import DataStructs
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import tqdm
import scipy.io as scio
import sys
from sklearn import preprocessing
from sklearn.decomposition import SparsePCA
import joblib

from feamap import basemap
from feamap import load_config
from feamap.preparation import drug_data, drug_fea, cellline_data, cellline_fea, fea_statistic, relationship_data, des_from_smiles, fgp_from_smiles, split_data, to_dist_matrix

prj_path = Path(__file__).parent.resolve()
save_path = prj_path / 'data' / 'processed_data'
save_path.mkdir(parents=True, exist_ok=True)
save_path_df = prj_path / 'data' / 'processed_data' / 'drug_fea' / 'map_transferred'
save_path_df.mkdir(parents=True, exist_ok=True)
save_path_cf = prj_path / 'data' / 'processed_data' / 'cellline_fea' / 'map_transferred'
save_path_cf.mkdir(parents=True, exist_ok=True)



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--ftype", type=str, default='fingerprint', choices=['fingerprint', 'descriptor', 'geneprofile'], help="")
    parser.add_argument("--datatype", type=str, default='cellline_fea', choices=['drug_fea', 'cellline_fea'], help="")
    parser.add_argument("--disttype", type=str, default='ALL', choices=['ALL'], help="the data source that the task working on")
    parser.add_argument("--fea_list", type=list, default=['MACCSFP', 'PharmacoErGFP', 'PubChemFP'],help="")
    parser.add_argument("--metric", type=str, default='cosine', choices=['correlation', 'cosine', 'jaccard'], help="")
    parser.add_argument("--fitmethod", type=str, default='umap', choices=['umap', 'tsne', 'mds'], help="")
    parser.add_argument("--channel", type=str, default='False', choices=['True', 'False'], help="")
    parser.add_argument("--split_data", type=str, default='False', choices=['True', 'False'], help="")
    parser.add_argument("--kfold_num", type=int, default=5, help="number of folds for K-Fold Cross-Validation, default 5")
    parser.add_argument("--scale_method", type=str, default='standard', choices=['standard', 'minmax', 'None'], help="")
    params = parser.parse_args()
    print(vars(params))


    # split train, valid and test
    if params.split_data == 'True':
        label = relationship_data()
        split_data(params, label, )

