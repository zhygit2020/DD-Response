import sys
from bisect import bisect
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
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict

from copy import copy
from sklearn.utils import shuffle 
import numpy as np
import pandas as pd
from model.metrics import evaluate,reshape_tf2th,to_categorical
from model.model import MultimapCNN, save_model, load_model
from feamap.basemap import load as load_map
import torch.nn.functional as F

prj_path = Path(__file__).parent.resolve().parent.resolve()

class run():
    def __init__(self, params):
        self.params = params

    def load_cvdata(self):
        cv_data = defaultdict(list)
        for k in range(self.params.kfold_num):
            train_k = pd.read_csv(prj_path / 'data' / 'processed_data' / 'split_cvdata' / f'{k}th_fold' / 'train_k.csv', index_col=0, header=0, low_memory=False)
            valid_k = pd.read_csv(prj_path / 'data' / 'processed_data' / 'split_cvdata' / f'{k}th_fold' / 'valid_k.csv', index_col=0, header=0, low_memory=False)
            test_k = pd.read_csv(prj_path / 'data' / 'processed_data' / 'split_cvdata' / f'{k}th_fold' / 'test_k.csv', index_col=0, header=0, low_memory=False)
            cv_data[k] = (train_k, valid_k, test_k)
        return cv_data
    def load_fea(self):
        fea_drug = np.load(prj_path / 'data' / 'processed_data' / 'drug_fea' / 'map_transferred' / 'drug_fea.npy').astype("float32")
        fea_cell = np.load(prj_path / 'data' / 'processed_data' / 'cellline_fea' / 'map_transferred' / 'cellline_fea.npy').astype("float32")
        print('fea_drug.shape: ', fea_drug.shape)
        print('fea_cell.shape: ', fea_cell.shape)
        id2idx_drug = pd.read_csv(prj_path / 'data' / 'processed_data' / 'drug_fea' / 'map_transferred' / 'drug_list.csv', index_col=1).iloc[:,0].to_dict()
        id2idx_cell = pd.read_csv(prj_path / 'data' / 'processed_data' / 'cellline_fea' / 'map_transferred' / 'cell_list.csv', index_col=1).iloc[:,0].to_dict()
        id2idx = dict()
        id2idx.update(id2idx_cell)
        id2idx.update(id2idx_drug)
        # print(id2idx)
        return fea_drug, fea_cell, id2idx

    def inits(self, fold, cv_data, id2idx, fea_drug, fea_cell):

        self.save_path = prj_path / 'pretrained' / f'{self.params.kfold_num}_fold_trainval' / f'batchsize_{self.params.batch_size}' / f'learningrate_{self.params.lr}' / f'monitor_{self.params.monitor}'
        
        (train, valid, test) = cv_data[fold]
        
        print(f'source: {self.params.source}')
        train = pd.concat([train.loc[train['source']==src] for src in self.params.source.split(",")]).sort_index()
        valid = pd.concat([valid.loc[valid['source']==src] for src in self.params.source.split(",")]).sort_index()
        test = pd.concat([test.loc[test['source']==src] for src in self.params.source.split(",")]).sort_index()
        train.to_csv(prj_path / 'data' / 'processed_data' / 'split_cvdata' / f'{fold}th_fold' / 'train_run_k.csv')
        valid.to_csv(prj_path / 'data' / 'processed_data' / 'split_cvdata' / f'{fold}th_fold' / 'valid_run_k.csv')
        test.to_csv(prj_path / 'data' / 'processed_data' / 'split_cvdata' / f'{fold}th_fold' / 'test_run_k.csv')

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

    def run(self,):
        cv_data = self.load_cvdata()
        fea_drug, fea_cell, id2idx = self.load_fea()
        allfold_train_data, allfold_val_data, allfold_test_data = {},{},{}
        assess = ['tn', 'fp', 'fn', 'tp', 'acc', 'auc', 'mcc', 'precision', 'recall', 'specificity', 'sensitivity', 'f1', 'prauc', 'av_prc']
        for fold in cv_data.keys():
            kfold_train_data, kfold_val_data, kfold_test_data = {},{},{}
            trainX, trainY, validX, validY, testX, testY = self.inits(fold, cv_data, id2idx, fea_drug, fea_cell)
            print(f'>>> working on fold {fold} <<<')
            clf = load_model(params = self.params, model_path = prj_path / 'pretrained' / f'{self.params.kfold_num}_fold_trainval' / f'batchsize_{self.params.batch_size}' / f'learningrate_{self.params.lr}' / f'monitor_{self.params.monitor}' / 'model' / f'{fold}th_fold', in_channels=(fea_drug.shape[-1],fea_cell.shape[-1]), gpuid=self.params.gpu)

            # fit finished
            trainY_pred, _latents_d, _latents_c = clf.run_loop(X=trainX,batch_size=self.params.batch_size)
            fprs, tprs, thresholds_auc, pres, recs, thresholds_prc, tn, fp, fn, tp, acc, auc, mcc, precision, recall, specificity, sensitivity, f1, prauc, av_prc = evaluate(y_true=to_categorical(num_classes = 2, y=trainY), y_pred=F.softmax(trainY_pred,dim=1))
            validY_pred, _latents_d, _latents_c = clf.run_loop(X=validX,batch_size=self.params.batch_size)
            fprs_val, tprs_val, thresholds_auc_val, pres_val, recs_val, thresholds_prc_val, tn_val, fp_val, fn_val, tp_val, acc_val, auc_val, mcc_val, precision_val, recall_val, specificity_val, sensitivity_val, f1_val, prauc_val, av_prc_val = evaluate(y_true=to_categorical(num_classes = 2, y=validY), y_pred=F.softmax(validY_pred,dim=1))
            testY_pred, _latents_d, _latents_c = clf.run_loop(X=testX,batch_size=self.params.batch_size)
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