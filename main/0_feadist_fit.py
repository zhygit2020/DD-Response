from rdkit import DataStructs
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import scipy.io as scio
import sys
from sklearn import preprocessing
from sklearn.decomposition import SparsePCA
import joblib
from feamap.preparation import drug_data, drug_fea, cellline_data, cellline_fea, fea_statistic, relationship_data, des_from_smiles, fgp_from_smiles, geneprofile_from_local, split_data, to_dist_matrix



prj_path = Path(__file__).parent.resolve()
save_path_df = prj_path / 'data' / 'processed_data' / 'drug_fea' / 'scale'
save_path_df.mkdir(parents=True, exist_ok=True)
save_path_cf = prj_path / 'data' / 'processed_data' / 'cellline_fea' / 'scale'
save_path_cf.mkdir(parents=True, exist_ok=True)

def MinMaxScaleClip(x, xmin, xmax):
    print("MinMaxScaleClip")
    scaled = (x - xmin) / ((xmax - xmin) + 1e-8)
    return scaled.clip(0, 1)

def StandardScaler(x, xmean, xstd):
    print("StandardScaler")
    return (x-xmean) / (xstd + 1e-8)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--drug_featype", type=str, default='fingerprint', choices=['descriptor', 'fingerprint', 'direct', 'map'], help="generate drug feature based on descriptor, fingerprint, map or directly")
    parser.add_argument("--scale_method", type=str, default='standard', choices=['standard', 'minmax', 'None'], help="")
    params = parser.parse_args()
    print(vars(params))

    # load data
    print('...loading data...')
    drugs = pd.read_csv(prj_path / 'data' / 'original_data' / 'scale' / 'all-data-merge-drug.csv', index_col=None, low_memory=False)
    # drugs.columns = ['drugid', 'cid']
    num_drugs = len(drugs.index)

    celllines = pd.read_csv(prj_path / 'data' / 'original_data' / 'scale' / 'rnaseq_tpm2-original-reactome-log2.csv', index_col=0, low_memory=False)
    # celllines.columns = ['gdsc-id', 'model_name']
    num_celllines = len(celllines.index)


    # featuregeneration
    # drug feature
    print('...working on drug feature...')
    drugs_fea, bitsinfo_drug = drug_fea(drugs, type=params.drug_featype)
    drugs_fea.to_csv(save_path_df / f'drug_molecular_{params.drug_featype}.csv')
    bitsinfo_drug.to_csv(save_path_df / f'drug_fea_bitsinfo_{params.drug_featype}.csv')

    # cellline feature
    print('...working on cellline feature...')
    celllines_fea, bitsinfo_cell = cellline_fea(celllines)
    celllines_fea.to_csv(save_path_cf / 'cellline_gene_geneprofile.csv')
    bitsinfo_cell.to_csv(save_path_cf / 'cell_fea_bitsinfo_geneprofile.csv')

    # statistic feature
    drugfea_scale = fea_statistic(drugs_fea)
    cellfea_scale = fea_statistic(celllines_fea)
    # save data
    drugfea_scale.to_pickle(save_path_df/f'{params.drug_featype}_scale.cfg')
    cellfea_scale.to_pickle(save_path_cf/'geneprofile_scale.cfg')


    drugs_to_dist, celllines_to_dist=[], []
    if params.scale_method == 'standard':
        for id, d_fea in drugs_fea.iterrows():
            drug_to_dist = StandardScaler(d_fea, drugfea_scale['mean'], drugfea_scale['std'])
            drugs_to_dist.append(np.nan_to_num(drug_to_dist))
        drugs_to_dist = np.stack(drugs_to_dist, axis=0)
        for id, c_fea in celllines_fea.iterrows():
            cellline_to_dist = StandardScaler(c_fea, cellfea_scale['mean'], cellfea_scale['std'])
            celllines_to_dist.append(np.nan_to_num(cellline_to_dist))
        celllines_to_dist = np.stack(celllines_to_dist, axis=0)

    elif params.scale_method == 'minmax':
        for id, d_fea in drugs_fea.iterrows():
            drug_to_dist = MinMaxScaleClip(d_fea, drugfea_scale['min'], drugfea_scale['max'])
            drugs_to_dist.append(np.nan_to_num(drug_to_dist))
        drugs_to_dist = np.stack(drugs_to_dist, axis=0)
        for id, c_fea in celllines_fea.iterrows():
            cellline_to_dist = MinMaxScaleClip(c_fea, cellfea_scale['min'], cellfea_scale['max'])
            celllines_to_dist.append(np.nan_to_num(cellline_to_dist))
        celllines_to_dist = np.stack(celllines_to_dist, axis=0)
        
    elif params.scale_method == 'None':
        drugs_to_dist = drugs_fea.values
        celllines_to_dist = celllines_fea.values
    # feature bits dist define
    pd.DataFrame(drugs_to_dist,columns=drugs_fea.columns,index=drugs_fea.index).to_csv(save_path_df / f'drug_normalizedfea_{params.drug_featype}.csv')
    pd.DataFrame(celllines_to_dist,columns=celllines_fea.columns,index=celllines_fea.index).to_csv(save_path_cf / 'cellline_normalized_geneprofile.csv')
    print('to_dist_matrix for drug')
    to_dist_matrix(drugs_to_dist, 'drug_fea', bitsinfo_drug['IDs'], params.drug_featype, methods = ['cosine', 'correlation', 'jaccard'])
    print('to_dist_matrix for cellline')
    to_dist_matrix(celllines_to_dist, 'cellline_fea', bitsinfo_cell['IDs'], 'geneprofile', methods = ['cosine', 'correlation', 'jaccard'])
    
    # should manually move the .cfg file to config directory before RGM construction








