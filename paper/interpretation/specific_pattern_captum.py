import sys
from pathlib import Path
prj_path = Path(__file__).parent.resolve().parent.resolve().parent.resolve() / 'main'
sys.path.append(str(prj_path))
import psutil
save_path = prj_path.parent.resolve() / 'paper' / 'interpretation'
save_path.mkdir(parents=True, exist_ok=True)

from model.metrics import evaluate,reshape_tf2th,to_categorical
from model.model import MultimapCNN, save_model, load_model
from feamap.basemap import load as load_map
from feamap.preparation import relationship_data
from collections import defaultdict
from tqdm import tqdm
from joblib import Parallel, delayed, load, dump
from copy import copy
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from rdkit import Chem
import argparse
import os
import pickle
sns.set(style='white',  font='sans-serif', font_scale=1.5)
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedShuffleSplit
import torch as th
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from captum.attr import IntegratedGradients


class MultimapCNN_dataset_2th(Dataset):
    def __init__(self, X, Y):
        super().__init__()
        X = (th.tensor(X[0]), th.tensor(X[1]))
        Y = th.tensor(Y)
        count = len(Y)
        self.src,  self.trg = [], []
        for i in range(count):
            self.src.append((X[0][i],X[1][i]))
            self.trg.append(Y[i])
    def __getitem__(self, index):
        return self.src[index], self.trg[index]
    def __len__(self):
        return len(self.src)

def load_tar():
    rep_drug_pairs = pd.read_excel(prj_path / 'data' / 'original_data' / 'targets' / 'Important-drug.xlsx', index_col=None)
    rep_cellline_pairs = pd.read_excel(prj_path / 'data' / 'original_data' / 'targets' / 'Important-cell-line.xlsx', index_col=None)
    rep_drug = rep_drug_pairs.drop_duplicates(subset='drugid').reset_index(drop=True)
    rep_cellline = rep_cellline_pairs.drop_duplicates(subset='gdsc-id').reset_index(drop=True)
    return rep_drug_pairs, rep_drug, rep_cellline_pairs, rep_cellline

def load_data(params, fold):
    # load fitted map and fitted model
    print('load drug map')
    mp_d = load_map(prj_path/ 'data' / 'processed_data' / 'fingerprint_fitted.mp')
    print('map loaded info: ', mp_d.ftype, '_', len(mp_d.flist), '_', mp_d.metric, '_', mp_d.method)
    print('load cellline map')
    mp_c = load_map(prj_path / 'data' / 'processed_data' / 'geneprofile_fitted.mp')
    print('map loaded info: ', mp_c.ftype, '_', len(mp_c.flist), '_', mp_c.metric, '_', mp_c.method)

    # load label
    cv_data = defaultdict(list)
    train_k = pd.read_csv(prj_path / 'data' / 'processed_data' / 'split_cvdata' / f'{fold}th_fold' / 'train_cv_k.csv', index_col=0, header=0, low_memory=False)
    valid_k = pd.read_csv(prj_path / 'data' / 'processed_data' / 'split_cvdata' / f'{fold}th_fold' / 'valid_cv_k.csv', index_col=0, header=0, low_memory=False)
    test_k = pd.read_csv(prj_path / 'data' / 'processed_data' / 'split_cvdata' / f'{fold}th_fold' / 'test_cv_k.csv', index_col=0, header=0, low_memory=False)
    cv_data = (train_k, valid_k, test_k)
    # load transferred fea
    fea_drug = np.load(prj_path / 'data' / 'processed_data' / 'drug_fea' / 'map_transferred' / 'drug_fea.npy').astype("float32")
    fea_cell = np.load(prj_path / 'data' / 'processed_data' / 'cellline_fea' / 'map_transferred' / 'cellline_fea.npy').astype("float32")
    print('fea_drug.shape: ', fea_drug.shape)
    print('fea_cell.shape: ', fea_cell.shape)

    id2idx_drug = pd.read_csv(prj_path / 'data' / 'processed_data' / 'drug_fea' / 'map_transferred' / 'drug_list.csv', index_col=1).iloc[:,0].to_dict()
    id2idx_cell = pd.read_csv(prj_path / 'data' / 'processed_data' / 'cellline_fea' / 'map_transferred' / 'cell_list.csv', index_col=1).iloc[:,0].to_dict()
    id2idx = dict()
    id2idx.update(id2idx_cell)
    id2idx.update(id2idx_drug)

    es_model = load_model(params = params, model_path = prj_path / 'pretrained' / f'{params.kfold_num}_fold_trainval' / f'batchsize_{params.batch_size}' / f'learningrate_{params.lr}' / f'monitor_{params.monitor}' / 'model' / f'{fold}th_fold', in_channels=(fea_drug.shape[-1],fea_cell.shape[-1]), gpuid=params.gpu)
    
    return mp_d, mp_c, es_model, cv_data, fea_drug, fea_cell, id2idx

def to_label(params, cv_data, set, fea_drug, fea_cell, tar, id2idx):
    # fold result
    (train_fold, valid_fold, test_fold) = cv_data
    print(f'source: {params.source}')
    train_fold = pd.concat([train_fold.loc[train_fold['source']==src] for src in params.source.split(",")]).sort_index()
    valid_fold = pd.concat([valid_fold.loc[valid_fold['source']==src] for src in params.source.split(",")]).sort_index()
    test_fold = pd.concat([test_fold.loc[test_fold['source']==src] for src in params.source.split(",")]).sort_index()
    # print(train_fold, valid_fold, test_fold)

    if not tar is None:
        if params.target=='rep_drug':
            train_fold = train_fold.loc[train_fold['drugid']==tar['drugid']]
            valid_fold = valid_fold.loc[valid_fold['drugid']==tar['drugid']]
            test_fold = test_fold.loc[test_fold['drugid']==tar['drugid']]
        elif params.target=='rep_cellline':
            train_fold = train_fold.loc[train_fold['gdsc-id']==tar['gdsc-id']]
            valid_fold = valid_fold.loc[valid_fold['gdsc-id']==tar['gdsc-id']]
            test_fold = test_fold.loc[test_fold['gdsc-id']==tar['gdsc-id']]
    else:
        pass

    if params.label=='positive':
        train_fold = train_fold.loc[train_fold['label']==1]
        valid_fold = valid_fold.loc[valid_fold['label']==1]
        test_fold = test_fold.loc[test_fold['label']==1]
    elif params.label=='negative':
        train_fold = train_fold.loc[train_fold['label']==0]
        valid_fold = valid_fold.loc[valid_fold['label']==0]
        test_fold = test_fold.loc[test_fold['label']==0]
    elif params.label=='all':
        pass

    data_drug_train = fea_drug[train_fold['drugid'].map(id2idx).values]
    data_cell_train = fea_cell[train_fold['gdsc-id'].map(id2idx).values]
    dataid_drug_train = train_fold['drugid'].values
    dataid_cell_train = train_fold['gdsc-id'].values
    
    data_drug_valid = fea_drug[valid_fold['drugid'].map(id2idx).values]
    data_cell_valid = fea_cell[valid_fold['gdsc-id'].map(id2idx).values]
    dataid_drug_valid = valid_fold['drugid'].values
    dataid_cell_valid = valid_fold['gdsc-id'].values
    
    data_drug_test = fea_drug[test_fold['drugid'].map(id2idx).values]
    data_cell_test = fea_cell[test_fold['gdsc-id'].map(id2idx).values]
    dataid_drug_test = test_fold['drugid'].values
    dataid_cell_test = test_fold['gdsc-id'].values

    print('reshape for torch')
    data_drug_train = reshape_tf2th(data_drug_train)
    data_cell_train = reshape_tf2th(data_cell_train)
    data_drug_valid = reshape_tf2th(data_drug_valid)
    data_cell_valid = reshape_tf2th(data_cell_valid)
    data_drug_test = reshape_tf2th(data_drug_test)
    data_cell_test = reshape_tf2th(data_cell_test)

    # split your data
    trainX = (data_drug_train, data_cell_train)
    idtrainX = (dataid_drug_train, dataid_cell_train)
    trainY = train_fold['label'].values
    validX = (data_drug_valid, data_cell_valid)
    idvalidX = (dataid_drug_valid, dataid_cell_valid)
    validY = valid_fold['label'].values
    testX = (data_drug_test, data_cell_test)
    idtestX = (dataid_drug_test, dataid_cell_test)
    testY = test_fold['label'].values
    allX = (np.concatenate((data_drug_train,data_drug_valid,data_drug_test),axis=0), np.concatenate((data_cell_train,data_cell_valid,data_cell_test),axis=0))
    idallX = (np.concatenate((dataid_drug_train,dataid_drug_valid,dataid_drug_test),axis=0), np.concatenate((dataid_cell_train,dataid_cell_valid,dataid_cell_test),axis=0))
    allY = np.concatenate((trainY,validY,testY),axis=0)

    if set == 'train' or set == 'train1' or set == 'train2' or set == 'train3':
        return trainX,trainY,idtrainX
    elif set == 'valid':
        return validX,validY,idvalidX
    elif set == 'test':
        return testX,testY,idtestX
    elif set == 'all':
        return allX,allY,idallX
    else:
        print('error')
        sys.exit()

def ForwardPropFeatureImp(params, model, X_true, id_true, Y_true, fea_drug, fea_cell, fold, tar, senres='sen'):

    print(f'Forward prop. Feature importance, IntegratedGradients of {senres}')

    posneg = 1 if senres=='sen' else 0

    model.zero_grad()
    model.to(th.device('cpu' if params.gpu == -1 else f'cuda:{params.gpu}'))
    model.eval()
    model.captum = True
    th.manual_seed(params.random_seed)
    np.random.seed(params.random_seed)
    ig = IntegratedGradients(model)

    data = MultimapCNN_dataset_2th(X_true,Y_true)
    dataloader = DataLoader(data, batch_size=32, shuffle=False)
    attr_senres = ([],[])
    delt_senres = []
    with tqdm(dataloader, ascii= True) as tq:
        for X, _ in tq:
            X = (X[0].to(th.device('cpu' if params.gpu == -1 else f'cuda:{params.gpu}')),X[1].to(th.device('cpu' if params.gpu == -1 else f'cuda:{params.gpu}')))
            attributions_senres, delta_senres = ig.attribute(X, target=posneg, return_convergence_delta=True, )
            attr_senres[0].append(attributions_senres[0].detach().cpu().numpy())
            attr_senres[1].append(attributions_senres[1].detach().cpu().numpy())
            delt_senres.append(delta_senres.detach().cpu().numpy())
    
    savepkl_path = save_path/'tar_impor'/f'{params.kfold_num}_fold_trainval'/f'{fold}th_fold'/ f'{params.target}' / 'on_all_patterns' / f'{params.label}'
    savepkl_path.mkdir(parents=True, exist_ok=True)
    with open(savepkl_path/f'rep_{tar}_drugattributions_{senres}_on_{params.set}data_monitor_{params.monitor}_fold{fold}.npy','wb') as f:
        np.save(f, np.concatenate(attr_senres[0], axis=0))
    with open(savepkl_path/f'rep_{tar}_celllineattributions_{senres}_on_{params.set}data_monitor_{params.monitor}_fold{fold}.npy','wb') as f:
        np.save(f, np.concatenate(attr_senres[1], axis=0))
    with open(savepkl_path/f'rep_{tar}_delta_{senres}_on_{params.set}data_monitor_{params.monitor}_fold{fold}.npy','wb') as f:
        np.save(f, np.concatenate(delt_senres, axis=0))
    
    return attr_senres

def traverse(params, fold, rep_drug, rep_cellline):
    
    mp_d, mp_c, clf, cv_data, fea_drug, fea_cell, id2idx = load_data(params, fold)
    mp_d.plot_grid(htmlpath=save_path)
    df_grid_d = mp_d.df_grid.sort_values(['y', 'x']).reset_index(drop=True)
    df_grid_d.to_csv(save_path/f'df_grid_drug.csv')
    mp_c.plot_grid(htmlpath=save_path)
    df_grid_c = mp_c.df_grid.sort_values(['y', 'x']).reset_index(drop=True)
    df_grid_c.to_csv(save_path/f'df_grid_cellline.csv')

    print("prepare for spesific pattern finding")
    lib = {}
    if params.target=='rep_drug':
        for idx, tar in rep_drug.iterrows():
            print(f">>>working on drug of {tar['drugid']}: {tar['drug_name']}<<<")
            X,Y,ID = to_label(params, cv_data, params.set, fea_drug, fea_cell, tar, id2idx)
            if X[0].shape[0]==Y.shape[0]==0:
                print('sample number should not be zero')
                continue
            dfp_s = ForwardPropFeatureImp(params, clf, X, ID, Y, fea_drug, fea_cell, fold, tar['drugid'], 'sen')
            dfp_r = ForwardPropFeatureImp(params, clf, X, ID, Y, fea_drug, fea_cell, fold, tar['drugid'], 'res')
            lib[tar['gdsc-id']]=(dfp_s,dfp_r)

    elif params.target=='rep_cellline':
        for idx, tar in rep_cellline.iterrows():
            print(f">>>working on cellline of {tar['gdsc-id']}<<<")
            X,Y,ID = to_label(params, cv_data, params.set, fea_drug, fea_cell, tar, id2idx)
            if X[0].shape[0]==Y.shape[0]==0:
                print('sample number should not be zero')
                continue
            dfp_s = ForwardPropFeatureImp(params, clf, X, ID, Y, fea_drug, fea_cell, fold, tar['gdsc-id'], 'sen')
            dfp_r = ForwardPropFeatureImp(params, clf, X, ID, Y, fea_drug, fea_cell, fold, tar['gdsc-id'], 'res')
            lib[tar['gdsc-id']]=(dfp_s,dfp_r)

    elif params.target=='rep_ALL':
        print(f">>>working on ALL pairs<<<")
        X,Y,ID = to_label(params, cv_data, params.set, fea_drug, fea_cell, None, id2idx)
        dfp_s = ForwardPropFeatureImp(params, clf, X, ID, Y, fea_drug, fea_cell, fold, 'ALL', 'sen')
        dfp_r = ForwardPropFeatureImp(params, clf, X, ID, Y, fea_drug, fea_cell, fold, 'ALL', 'res')
        lib['ALL']=(dfp_s,dfp_r)

    return lib

def bits_scores(params, fold, bits, senres='sen'):
    for bit in bits:
        print(f'>>>>>>bit_score_of_{bit}_of_all_folds<<<<<<')
        bit_score = {}
        df_grid = pd.read_csv(save_path/f"df_grid_{params.interpret_on}.csv", index_col=0)

        mp_d, mp_c, clf, cv_data, fea_drug, fea_cell, id2idx = load_data(params, fold)
        X_true,Y_true,id_true = to_label(params, cv_data, params.set, fea_drug, fea_cell, None, id2idx)
        
        channels = mp_d.channel_col if params.interpret_on=='drug' else mp_c.channel_col

        savepkl_path = save_path/'tar_impor'/f'{params.kfold_num}_fold_trainval'/f'{fold}th_fold'/ f'rep_ALL' / 'on_all_patterns' / f'{params.label}'
        with open(savepkl_path/f'rep_ALL_{params.interpret_on}attributions_{senres}_on_{params.set}data_monitor_{params.monitor}_fold{fold}.npy','rb') as f:
            attributions_senres = np.load(f)

        ts = df_grid.loc[df_grid['v']==bit]
        y = ts.y.values.squeeze(-1)
        x = ts.x.values.squeeze(-1)
        v = ts.v.values.squeeze(-1)
        dict_v = {'Did':id_true[0], 'Cid':id_true[1], 'Y_true':Y_true,}

        # for channel in range(attributions_sen[0].shape[1]):
        for i, channel in enumerate(channels):
            dict_v[f'IG_score_sen_{i}th_channel_{channel}'] = attributions_senres[:,i,y,x]
        # sum all channels
        IG_score_senres = np.sum(attributions_senres[:,:,y,x], axis=1)
            
        dict_v[f'IG_score_{senres}'] = IG_score_senres
        
        r = pd.DataFrame.from_dict(dict_v).set_index(['Did', 'Cid', 'Y_true'])

        savetraverse_path = save_path/'tar_impor'/f'{params.kfold_num}_fold_trainval'/f'{fold}th_fold'/f'rep_ALL'/f'on_{params.interpret_on}'/ f'{params.label}'
        savetraverse_path.mkdir(parents=True, exist_ok=True)
        r.to_csv(savetraverse_path/f"rep_ALL_{params.interpret_on}Feature_importance_of_{v}_{senres}_on_{params.set}data_monitor_{params.monitor}_fold{fold}.csv")
        
        bit_score[fold] = r


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=-1, help="GPU id, -1 for cpu")
    parser.add_argument("--kfold_num", type=int, default=5, help="number of folds for K-Fold Cross-Validation, default 5")
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="model batch size")
    parser.add_argument("--fold", type=int, default=0, help="fold NO. of the target model")
    parser.add_argument("--interpret_on", type=str, default='cellline', choices=['all', 'drug', 'cellline'], help="interpretation on all, drug or cellline")
    parser.add_argument("--monitor", type=str, default='loss_val', choices=['loss_val', 'acc_val'], help="earlystop monitor")
    parser.add_argument("--target", type=str, default='rep_ALL', choices=['rep_drug', 'rep_cellline', 'rep_ALL'], help="the representative drugs or celllined to be investigated") #, it should be opposite to {--interpret_on}???
    parser.add_argument("--set", type=str, default='train', choices=['train', 'valid', 'test','train1','train2','train3','all'], help="datasets set")
    parser.add_argument("--label", type=str, default='all', choices=['all','positive', 'negative'], help="interpret on positive or negative or all interactions")

    parser.add_argument("--kfold_cv_type", type=str, default='pair', choices=['pair', 'drug', 'cellline'], help="K-Fold Cross-Validation dataset splitting choice")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight for L2 loss") 
    parser.add_argument("--n_epochs", type=int, default=512, help="number of training epochs")
    parser.add_argument("--n_attn_heads", type=int, default=1, help="num of attention heads at layer")
    parser.add_argument("--attn_units", type=int, default=8, help="attention_size at each attention head, its length equals n_attn_heads")
    parser.add_argument("--task", type=str, default='cv', choices=['cv', 'test'], help="model work mode, cross-validating or testing")
    parser.add_argument("--metric", type=str, default='ACC', choices=['ACC', 'ROC'], help="optimaizer metric")
    parser.add_argument("--source", type=str, default='GDSC,CTRP,CCLE', help="the data source that the task working on")


    params = parser.parse_args()
    print(vars(params))

    # load and init data
    rep_drug_pairs, rep_drug, rep_cellline_pairs, rep_cellline = load_tar()
    print(f"target drugs: {rep_drug['drugid'].values}")
    print(f"target celllines: {rep_cellline['gdsc-id'].values}")

    fold_lib = {}

    fold = params.fold
    print(f"********************WORKING ON FOLD {fold}********************")
    fold_lib[fold] = traverse(params, fold, rep_drug, rep_cellline)

