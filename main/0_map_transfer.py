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

def fit_map(disttype = 'ALL', datatype = 'drug_fea', ftype = 'fingerprint', flist=None, split_channels=True, metric = 'cosine', fitmethod = 'umap'):
    mp = basemap.Map(disttype = disttype, datatype = datatype, ftype = ftype, flist = flist, fmap_type='grid', split_channels=split_channels,  metric=metric, var_thr=1e-4)
    mp.fit(method = fitmethod)
    # Visulization and save your fitted map
    mp.plot_grid(htmlpath=save_path, htmlname=None)
    mp.save(save_path / f'{ftype}_fitted.mp')
    return mp

def load_map(disttype = 'ALL', datatype = 'drug_fea', ftype = 'fingerprint', flist=None, split_channels=True, metric = 'cosine', fitmethod = 'umap'):
    mp = basemap.Map(disttype = disttype, datatype = datatype, ftype = ftype, flist = flist, fmap_type='grid', split_channels=split_channels,  metric=metric, var_thr=1e-4)
    mp = mp.load(save_path / f'{ftype}_fitted.mp')
    # Visulization and save your fitted map
    mp.plot_grid(htmlpath=save_path, htmlname=None)
    return mp

def trans_map(params, data_df, mp, datatype = 'drug_fea'):
    if datatype == 'drug_fea':
        fea_o, bitsinfo_drug = drug_fea(data_df, type=params.ftype)
        fea_o.to_csv(save_path_df / f'drug_molecular_{params.ftype}_beforetrans.csv')
        bitsinfo_drug.to_csv(save_path_df / f'drug_fea_bitsinfo_{params.ftype}_beforetrans.csv')
    elif datatype == 'cellline_fea':
        fea_o, bitsinfo_cell = cellline_fea(data_df)
        fea_o.to_csv(save_path_cf / 'cellline_gene_geneprofile_beforetrans.csv')
        bitsinfo_cell.to_csv(save_path_cf / 'cell_fea_bitsinfo_geneprofile_beforetrans.csv')
        
    fea_map, ids = mp.batch_transform(arrs=fea_o, scale=True, scale_method=params.scale_method)

    print('fea_map.shape', fea_map.shape)
    return fea_map.astype('float32'), ids






if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--ftype", type=str, default='fingerprint', choices=['fingerprint', 'descriptor', 'geneprofile'], help="")
    parser.add_argument("--datatype", type=str, default='cellline_fea', choices=['drug_fea', 'cellline_fea'], help="")
    parser.add_argument("--disttype", type=str, default='ALL', choices=['ALL'], help="the distance matrix source that the task working on")
    parser.add_argument("--fea_list", type=list, default=['MACCSFP', 'PharmacoErGFP', 'PubChemFP'],help="")
    parser.add_argument("--metric", type=str, default='cosine', choices=['correlation', 'cosine', 'jaccard'], help="")
    parser.add_argument("--fitmethod", type=str, default='umap', choices=['umap', 'tsne', 'mds'], help="")
    parser.add_argument("--channel", type=str, default='False', choices=['True', 'False'], help="")
    parser.add_argument("--kfold_num", type=int, default=5, help="number of folds for K-Fold Cross-Validation, default 5")
    parser.add_argument("--scale_method", type=str, default='standard', choices=['standard', 'minmax', 'None'], help="")
    params = parser.parse_args()
    print(vars(params))


    if params.datatype == 'drug_fea':
        # load map
        mp_d = fit_map(disttype = params.disttype, datatype = 'drug_fea', ftype = params.ftype, flist=params.fea_list, split_channels=True, metric = params.metric, fitmethod = params.fitmethod)
        # mp_d = load_map(disttype = params.disttype, datatype = 'drug_fea', ftype = params.ftype, flist=params.fea_list, split_channels=False, metric = params.metric, fitmethod = params.fitmethod)
        drugs, num_drugs = drug_data()
        drugmap, ids = trans_map(params, drugs, mp_d, datatype = 'drug_fea')
        # save data
        with open(save_path_df/'drug_fea.npy', 'wb') as f:
            np.save(f, drugmap)
        pd.DataFrame(ids).to_csv(save_path_df/'drug_list.csv')

    elif params.datatype == 'cellline_fea':
        # load map
        mp_c = fit_map(disttype = params.disttype, datatype = 'cellline_fea', ftype = 'geneprofile', flist=[], split_channels=True, metric = params.metric, fitmethod = params.fitmethod)
        # mp_c = load_map(disttype = params.disttype, datatype = 'cellline_fea', ftype = 'geneprofile', flist=[], split_channels=False, metric = params.metric, fitmethod = params.fitmethod)
        celllines, num_celllines = cellline_data()
        cellmap, ids = trans_map(params, celllines, mp_c, datatype = 'cellline_fea')
        # save data
        with open(save_path_cf/'cellline_fea.npy', 'wb') as f:
            np.save(f, cellmap)
        pd.DataFrame(ids).to_csv(save_path_cf/'cell_list.csv')
    

