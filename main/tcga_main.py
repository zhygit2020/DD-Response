import sys
from bisect import bisect
import argparse
from rdkit import DataStructs
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import numpy as np
import pandas as pd
from pathlib import Path
from sqlalchemy import column
from tqdm import tqdm
import scipy.io as scio
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from collections import defaultdict
import time
import os
import psutil
from copy import copy
from sklearn.utils import shuffle 
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from feamap import basemap
from feamap.preparation import fea_statistic, relationship_data, des_from_smiles, fgp_from_smiles, geneprofile_from_local, drug_fea, cellline_fea
from model.metrics import evaluate,reshape_tf2th,to_categorical
from model.model import MultimapCNN, MultimapCNN_dataset, EarlyStopping, save_model, load_model

current_path = Path(__file__).parent.resolve()
prj_path = current_path / 'transfer'
save_path = prj_path / 'data' / 'processed_data'
save_path.mkdir(parents=True, exist_ok=True)
save_path_df = prj_path / 'data' / 'processed_data' / 'drug_fea' / 'map_transferred'
save_path_df.mkdir(parents=True, exist_ok=True)
save_path_cf = prj_path / 'data' / 'processed_data' / 'cellline_fea' / 'map_transferred'
save_path_cf.mkdir(parents=True, exist_ok=True)


class transfer():
    def __init__(self, params):
        self.params = params
        self.patience = 16
        self.tcga_pair = pd.read_csv(prj_path / 'data' / 'original_data' / 'patient-drug.csv', low_memory=False)
        self.transfer_drugs = pd.read_csv(prj_path / 'data' / 'original_data' / 'drug.csv', index_col=None, low_memory=False).drop_duplicates(subset=['drugid'])
        transfer_celllines = pd.read_csv(prj_path / 'data' / 'original_data' / 'patient.csv', index_col=0, low_memory=False)
        celllines_o = pd.read_csv(current_path / 'data' / 'original_data' / 'all-data-merge-cell.csv', index_col=0, low_memory=False)
        transfer_celllines = transfer_celllines[celllines_o.columns]
        NAgenes = transfer_celllines.T.loc[transfer_celllines.T.iloc[:,1]=='NA'].index.tolist()
        for g in NAgenes:
            transfer_celllines[g] = celllines_o[g].mean()
        transfer_celllines.to_csv(prj_path / 'data' / 'original_data' / 'patient_fillna.csv')
        self.transfer_celllines = transfer_celllines

        self.mp_d = self.load_map(disttype = params.disttype, datatype = 'drug_fea', ftype = params.ftype, flist=['MACCSFP', 'PharmacoErGFP', 'PubChemFP'], split_channels=True, metric = 'cosine', fitmethod = params.fitmethod)
        self.mp_c = self.load_map(disttype = params.disttype, datatype = 'cellline_fea', ftype = 'geneprofile', flist=[], split_channels=True, metric = 'cosine', fitmethod = params.fitmethod)


    def split_cvtdata(self, pair):
        cvt_data = defaultdict(list)
        skf = StratifiedKFold(n_splits=self.params.kfold_num, shuffle=True, random_state = self.params.random_seed)
        for k in range(self.params.kfold_num):
            cvt_data[k]=(pd.read_csv(current_path / 'data' / 'processed_data' / 'split_cvdata' / f'{k}th_fold' / 'train_k.csv', index_col=0), pd.read_csv(current_path / 'data' / 'processed_data' / 'split_cvdata' / f'{k}th_fold' / 'valid_k.csv', index_col=0), pd.read_csv(current_path / 'data' / 'processed_data' / 'split_cvdata' / f'{k}th_fold' / 'test_k.csv', index_col=0))
            savecv_path = prj_path / 'data' / 'processed_data' / 'split_cvdata' / f'{k}th_fold'
            savecv_path.mkdir(parents=True, exist_ok=True)
            cvt_data[k][0].to_csv(savecv_path / 'train_k.csv' )
            cvt_data[k][1].to_csv(savecv_path / 'valid_k.csv' )
            cvt_data[k][2].to_csv(savecv_path / 'test_k.csv' )
        return cvt_data
    
    def load_map(self, disttype = 'ALL', datatype = 'drug_fea', ftype = 'fingerprint', flist=None, split_channels=True, metric = 'cosine', fitmethod = 'umap'):
        mp_path = current_path / 'data' / 'processed_data'
        mp = basemap.Map(disttype = disttype, datatype = datatype, ftype = ftype, flist = flist, fmap_type='grid', split_channels=split_channels,  metric=metric, var_thr=1e-4)
        mp = mp.load(mp_path / f'{ftype}_fitted.mp')
        # Visulization and save your fitted map
        mp.plot_grid(htmlpath=mp_path, htmlname=None)
        return mp

    def trans_map(self, params, data_df, mp, datatype = 'drug_fea'):
        if datatype == 'drug_fea':
            fea_o, bitsinfo_drug = drug_fea(data_df, type=params.ftype)
            fea_o.to_csv(save_path_df / f'drug_molecular_{params.ftype}_beforetrans.csv')
            bitsinfo_drug.to_csv(save_path_df / f'drug_fea_bitsinfo_{params.ftype}_beforetrans.csv')
            fea_map, ids = mp.batch_transform(arrs=fea_o, scale=True, scale_method=params.scale_method)
            with open(save_path_df/'drug_fea.npy', 'wb') as f:
                np.save(f, fea_map)
            pd.DataFrame(ids).to_csv(save_path_df/'drug_list.csv')
        elif datatype == 'cellline_fea':
            fea_o, bitsinfo_cell = cellline_fea(data_df)
            fea_o.to_csv(save_path_cf / 'cellline_gene_geneprofile_beforetrans.csv')
            bitsinfo_cell.to_csv(save_path_cf / 'cell_fea_bitsinfo_geneprofile_beforetrans.csv')
            fea_map, ids = mp.batch_transform(arrs=fea_o, scale=True, scale_method='None')
            with open(save_path_cf/'cellline_fea.npy', 'wb') as f:
                np.save(f, fea_map)
            pd.DataFrame(ids).to_csv(save_path_cf/'cell_list.csv')
            
        print('fea_map.shape', fea_map.shape)
        return fea_map.astype('float32'), ids

    def feamap_generation(self, drugs, celllines):
        fea_drug, ids_d = self.trans_map(self.params, drugs, self.mp_d, datatype = 'drug_fea')
        fea_cell, ids_c = self.trans_map(self.params, celllines, self.mp_c, datatype = 'cellline_fea')
        print('fea_drug.shape: ', fea_drug.shape)
        print('fea_cell.shape: ', fea_cell.shape)
        id2idx_drug = pd.DataFrame(ids_d).reset_index().set_index(0).iloc[:,0].to_dict()
        id2idx_cell = pd.DataFrame(ids_c).reset_index().set_index(0).iloc[:,0].to_dict()
        id2idx = dict()
        id2idx.update(id2idx_cell)
        id2idx.update(id2idx_drug)
        return fea_drug, fea_cell, id2idx

    def inits(self, fold, cv_data, id2idx, fea_drug, fea_cell):

        self.save_path = prj_path / 'pretrained' / f'TCGA_transfer' / f'{self.params.kfold_num}_fold_trainval' / f'batchsize_{self.params.batch_size}' / f'learningrate_{self.params.lr}' / f'monitor_{self.params.monitor}'
        
        (train, valid, test) = cv_data[fold]

        data_drug_train = fea_drug[train['drugid'].map(id2idx).values]
        data_cell_train = fea_cell[train['gdsc-id'].map(id2idx).values]
        data_drug_valid = fea_drug[valid['drugid'].map(id2idx).values]
        data_cell_valid = fea_cell[valid['gdsc-id'].map(id2idx).values]
        data_drug_test = fea_drug[test['drugid'].map(id2idx).values]
        data_cell_test = fea_cell[test['gdsc-id'].map(id2idx).values]
        # reshape for torch 
        print('reshape for torch')
        data_drug_train = reshape_tf2th(data_drug_train)
        data_cell_train = reshape_tf2th(data_cell_train)
        data_drug_valid = reshape_tf2th(data_drug_valid)
        data_cell_valid = reshape_tf2th(data_cell_valid)
        data_drug_test = reshape_tf2th(data_drug_test)
        data_cell_test = reshape_tf2th(data_cell_test)
        
        # split your data
        trainX = (data_drug_train, data_cell_train)
        trainY = train['label'].values
        validX = (data_drug_valid, data_cell_valid)
        validY = valid['label'].values
        testX = (data_drug_test, data_cell_test)
        testY = test['label'].values

        return trainX, trainY, validX, validY, testX, testY

    def fit(self, model, save_path, trainX, trainY, validX, validY):
        print(model)
        # reshape
        training_data = MultimapCNN_dataset(trainX, trainY)
        valid_data = MultimapCNN_dataset(validX, validY)
        train_dataloader = DataLoader(training_data, batch_size=self.params.batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_data, batch_size=self.params.batch_size, shuffle=True)
        loss_fn = nn.CrossEntropyLoss() # have softmax layer
        optimizer = th.optim.Adam(model.parameters(), lr=self.params.lr, weight_decay=1e-8)
        early_stopping = EarlyStopping(patience=self.patience, verbose=True, monitor = self.params.monitor)
        for t in range(self.params.n_epochs):
            time_epstart = time.time()
            print(f"-------------------------------\nEpoch {t+1}\n-------------------------------")
            train_loss, train_logits, train_label = model.train_loop(train_dataloader, loss_fn, optimizer)
            valid_loss, valid_logits, valid_label = model.test_loop(valid_dataloader, loss_fn, )
            train_fprs, train_tprs, train_thresholds_auc, train_pres, train_recs, train_thresholds_prc, train_tn, train_fp, train_fn, train_tp, train_acc, train_auc, train_mcc, train_precision, train_recall, train_specificity, train_sensitivity, train_f1, train_prauc, train_av_prc = evaluate(y_true=to_categorical(num_classes = 2, y=train_label), y_pred=F.softmax(train_logits,dim=1))
            valid_fprs, valid_tprs, valid_thresholds_auc, valid_pres, valid_recs, valid_thresholds_prc, valid_tn, valid_fp, valid_fn, valid_tp, valid_acc, valid_auc, valid_mcc, valid_precision, valid_recall, valid_specificity, valid_sensitivity, valid_f1, valid_prauc, valid_av_prc = evaluate(y_true=to_categorical(num_classes = 2, y=valid_label), y_pred=F.softmax(valid_logits,dim=1))
            print(f'Epoch {t+1} result: valid_loss={valid_loss:.4f}, valid_acc={valid_acc:.4f}, valid_auc={valid_auc:.4f}, valid_mcc={valid_mcc:.4f}, valid_precision={valid_precision:.4f}, valid_recall={valid_recall:.4f}, valid_specificity={valid_specificity:.4f}, valid_sensitivity={valid_sensitivity:.4f}, valid_f1={valid_f1:.4f}, valid_prauc={valid_prauc:.4f}')
            early_stopping(score = {'loss_val':valid_loss, 'acc_val':valid_acc}, model = model, model_path = save_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    def run(self,):
        cv_data = self.split_cvtdata(self.tcga_pair)
        fea_drug, fea_cell, id2idx = self.feamap_generation(self.transfer_drugs,self.transfer_celllines)
        allfold_train_data, allfold_val_data, allfold_test_data = {},{},{}
        assess = ['tn', 'fp', 'fn', 'tp', 'acc', 'auc', 'mcc', 'precision', 'recall', 'specificity', 'sensitivity', 'f1', 'prauc', 'av_prc']
        
        for fold in cv_data.keys():
            kfold_train_data, kfold_val_data, kfold_test_data = {},{},{}
            trainX, trainY, validX, validY, testX, testY = self.inits(fold, cv_data, id2idx, fea_drug, fea_cell)
            # save dpath
            save_path_model = self.save_path / 'model' / f'{fold}th_fold'
            save_path_model.mkdir(parents=True, exist_ok=True)
            # fit your model
            print(f'>>> working on fold {fold} <<<')
            print(f'**********fitting**********')
            clf = MultimapCNN(self.params, in_channels=(fea_drug.shape[-1],fea_cell.shape[-1]))
            self.fit(clf, save_path_model, trainX, trainY, validX, validY)
            
            es_model = load_model(params=self.params, model_path=save_path_model, in_channels=(fea_drug.shape[-1],fea_cell.shape[-1]) , gpuid=self.params.gpu)
            # fit finished
            trainY_pred, _latents_d, _latents_c = es_model.run_loop(X=trainX,batch_size=self.params.batch_size)
            fprs, tprs, thresholds_auc, pres, recs, thresholds_prc, tn, fp, fn, tp, acc, auc, mcc, precision, recall, specificity, sensitivity, f1, prauc, av_prc = evaluate(y_true=to_categorical(num_classes = 2, y=trainY), y_pred=F.softmax(trainY_pred,dim=1))
            validY_pred, _latents_d, _latents_c = es_model.run_loop(X=validX,batch_size=self.params.batch_size)
            fprs_val, tprs_val, thresholds_auc_val, pres_val, recs_val, thresholds_prc_val, tn_val, fp_val, fn_val, tp_val, acc_val, auc_val, mcc_val, precision_val, recall_val, specificity_val, sensitivity_val, f1_val, prauc_val, av_prc_val = evaluate(y_true=to_categorical(num_classes = 2, y=validY), y_pred=F.softmax(validY_pred,dim=1))
            testY_pred, _latents_d, _latents_c = es_model.run_loop(X=testX,batch_size=self.params.batch_size)
            fprs_test, tprs_test, thresholds_auc_test, pres_test, recs_test, thresholds_prc_test, tn_test, fp_test, fn_test, tp_test, acc_test, auc_test, mcc_test, precision_test, recall_test, specificity_test, sensitivity_test, f1_test, prauc_test, av_prc_test = evaluate(y_true=to_categorical(num_classes = 2, y=testY), y_pred=F.softmax(testY_pred,dim=1))
            print(f'-------------------------------- finish {fold} fold cv --------------------------------')
            print(f'TRAIN result: acc = {acc:.4f}; auc = {auc:.4f}, mcc = {mcc:.4f}, precision = {precision:.4f}, recall = {recall:.4f}, specificity = {specificity:.4f}, sensitivity = {sensitivity:.4f}, f1 = {f1:.4f}, prauc = {prauc:.4f}, av_prc = {av_prc:.4f}')

            print(f'VALID result: acc = {acc_val:.4f}; auc = {auc_val:.4f}, mcc = {mcc_val:.4f}, precision = {precision_val:.4f}, recall = {recall_val:.4f}, specificity = {specificity_val:.4f}, sensitivity = {sensitivity_val:.4f}, f1 = {f1_val:.4f}, prauc = {prauc_val:.4f}, av_prc = {av_prc_val:.4f}')

            print(f'TEST result: acc = {acc_test:.4f}; auc = {auc_test:.4f}, mcc = {mcc_test:.4f}, precision = {precision_test:.4f}, recall = {recall_test:.4f}, specificity = {specificity_test:.4f}, sensitivity = {sensitivity_test:.4f}, f1 = {f1_test:.4f}, prauc = {prauc_test:.4f}, av_prc = {av_prc_test:.4f}')

            for ass in assess:
                exec(f"kfold_train_data['{ass}'] = {ass}")
            for ass in assess:
                exec(f"kfold_val_data['{ass}'] = {ass}_val")
            for ass in assess:
                exec(f"kfold_test_data['{ass}'] = {ass}_test")
            allfold_train_data[fold] = kfold_train_data
            allfold_val_data[fold] = kfold_val_data
            allfold_test_data[fold] = kfold_test_data

            ROC_savepath = self.save_path / 'ROC_data' / f'{fold}th_fold'
            ROC_savepath.mkdir(parents=True, exist_ok=True)
            pd.DataFrame.from_dict({'fprs':fprs, 'tprs':tprs, 'thresholds':thresholds_auc}).to_csv(ROC_savepath / f'train_ROC_for_{fold}th_fold.csv')
            pd.DataFrame.from_dict({'fprs':fprs_val, 'tprs':tprs_val, 'thresholds':thresholds_auc_val}).to_csv(ROC_savepath / f'val_ROC_for_{fold}th_fold.csv')
            pd.DataFrame.from_dict({'fprs':fprs_test, 'tprs':tprs_test, 'thresholds':thresholds_auc_test}).to_csv(ROC_savepath / f'test_ROC_for_{fold}th_fold.csv')
            
            PRC_savepath = self.save_path / 'PRC_data' / f'{fold}th_fold'
            PRC_savepath.mkdir(parents=True, exist_ok=True)
            pd.DataFrame.from_dict({'pres':pres, 'recs':recs, 'thresholds':thresholds_prc}).to_csv(PRC_savepath / f'train_PRC_for_{fold}th_fold.csv')
            pd.DataFrame.from_dict({'pres':pres_val, 'recs':recs_val, 'thresholds':thresholds_prc_val}).to_csv(PRC_savepath / f'val_PRC_for_{fold}th_fold.csv')
            pd.DataFrame.from_dict({'pres':pres_test, 'recs':recs_test, 'thresholds':thresholds_prc_test}).to_csv(PRC_savepath / f'test_PRC_for_{fold}th_fold.csv')

            logits_savepath = self.save_path / 'logits_data' / f'{fold}th_fold'
            logits_savepath.mkdir(parents=True, exist_ok=True)
            pd.DataFrame.from_dict({'y_true':trainY, 'y_pred':trainY_pred}).to_csv(logits_savepath / f'train_logits_{fold}th_fold.csv')
            pd.DataFrame.from_dict({'y_true':validY, 'y_pred':validY_pred}).to_csv(logits_savepath / f'val_logits_{fold}th_fold.csv')
            pd.DataFrame.from_dict({'y_true':testY, 'y_pred':testY_pred}).to_csv(logits_savepath / f'test_logits_{fold}th_fold.csv')

        ave_tr, ave_v, ave_te = self.save(assess, allfold_train_data, allfold_val_data, allfold_test_data)
        print(f'-------------------------------- {self.params.kfold_num} folds average result --------------------------------')
        print(f'AVERAGE_TRAIN result: acc = {ave_tr.acc:.4f}; auc = {ave_tr.auc:.4f}, mcc = {ave_tr.mcc:.4f}, precision = {ave_tr.precision:.4f}, recall = {ave_tr.recall:.4f}, specificity = {ave_tr.specificity:.4f}, sensitivity = {ave_tr.sensitivity:.4f}, f1 = {ave_tr.f1:.4f}, prauc = {ave_tr.prauc:.4f}, av_prc = {ave_tr.av_prc:.4f}')
        print(f'AVERAGE_VALID result: acc = {ave_v.acc:.4f}; auc = {ave_v.auc:.4f}, mcc = {ave_v.mcc:.4f}, precision = {ave_v.precision:.4f}, recall = {ave_v.recall:.4f}, specificity = {ave_v.specificity:.4f}, sensitivity = {ave_v.sensitivity:.4f}, f1 = {ave_v.f1:.4f}, prauc = {ave_v.prauc:.4f}, av_prc = {ave_v.av_prc:.4f}')
        print(f'AVERAGE_TEST result: acc = {ave_te.acc:.4f}; auc = {ave_te.auc:.4f}, mcc = {ave_te.mcc:.4f}, precision = {ave_te.precision:.4f}, recall = {ave_te.recall:.4f}, specificity = {ave_te.specificity:.4f}, sensitivity = {ave_te.sensitivity:.4f}, f1 = {ave_te.f1:.4f}, prauc = {ave_te.prauc:.4f}, av_prc = {ave_te.av_prc:.4f}')

    def save(self, assess, allfold_train_data, allfold_val_data, allfold_test_data):
        result_tr = pd.DataFrame([])
        result_v = pd.DataFrame([])
        result_te = pd.DataFrame([])
        for ass in assess:
            for k in range(self.params.kfold_num):
                result_tr.at[k,'Foldid'] = k
                result_v.at[k,'Foldid'] = k
                result_te.at[k,'Foldid'] = k
                result_tr.at[k,ass] = allfold_train_data[k][ass]
                result_v.at[k,ass] = allfold_val_data[k][ass]
                result_te.at[k,ass] = allfold_test_data[k][ass]
                
        resultdata_savepath = self.save_path / 'result_data'
        resultdata_savepath.mkdir(parents=True, exist_ok=True)
        result_tr.to_csv(resultdata_savepath / 'train_result.csv')
        result_v.to_csv(resultdata_savepath / 'val_result.csv')
        result_te.to_csv(resultdata_savepath / 'test_result.csv')

        return result_tr.mean(), result_v.mean(), result_te.mean()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--kfold_num", type=int, default=5, help="number of folds for K-Fold Cross-Validation, default 5")
    parser.add_argument("--kfold_cv_type", type=str, default='pair', choices=['pair', 'drug', 'cellline'], help="K-Fold Cross-Validation dataset splitting choice")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU id, -1 for cpu")
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight for L2 loss") 
    parser.add_argument("--n_epochs", type=int, default=512, help="number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="model batch size")
    parser.add_argument("--disttype", type=str, default='ALL', choices=['ALL'], help="the data source that the task working on")
    parser.add_argument("--ftype", type=str, default='fingerprint', choices=['fingerprint', 'descriptor', 'geneprofile'], help="")
    parser.add_argument("--fitmethod", type=str, default='umap', choices=['umap', 'tsne', 'mds'], help="")
    parser.add_argument("--n_attn_heads", type=int, default=1, help="num of attention heads at layer")
    parser.add_argument("--attn_units", type=int, default=8, help="attention_size at each attention head, its length equals n_attn_heads")
    parser.add_argument("--monitor", type=str, default='loss_val', choices=['loss_val', 'acc_val'], help="earlystop monitor")
    parser.add_argument("--metric", type=str, default='ACC', choices=['ACC', 'ROC'], help="optimaizer metric")
    parser.add_argument("--source", type=str, default='GDSC,CTRP,CCLE', help="the data source that the task working on")
    parser.add_argument("--scale_method", type=str, default='standard', choices=['standard', 'minmax', 'None'], help="")



    params = parser.parse_args()
    print(vars(params))

    runner = transfer(params)
    runner.run()


