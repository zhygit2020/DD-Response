import copy
import numpy as np
from sklearn import metrics
# import tensorflow as tf
import time
import psutil
import os

def reshape_tf2th( X):
    # N, W, H, C = X.shape
    # print(f'feamap shape before reshape: N, W, H, C = {N, W, H, C}')
    X = np.moveaxis(X,3,1)
    # X = np.transpose(X,(0,3,1,2))
    print(f'feamap shape after reshape: N, C, W, H = {X.shape}')
    return X

def to_categorical(num_classes, y):
    return np.eye(num_classes, dtype=np.long)[y]

def evaluate(y_true, y_pred):
    y_label = y_true[:,-1]
    y_score = y_pred[:,-1]
    # ROC, AUC
    fprs, tprs, thresholds_auc = metrics.roc_curve(y_label, y_score)
    auc = metrics.auc(fprs, tprs)
    # PRAUC
    pres, recs, thresholds_prc = metrics.precision_recall_curve(y_label, y_score)
    prauc = metrics.auc(recs, pres)
    av_prc = metrics.average_precision_score(y_label, y_score)
    # scores' label prediction by threshold
    threshold = 0.5
    label_pred = copy.deepcopy(y_score)
    label_pred = np.where(y_score >= threshold, np.ones_like(label_pred), label_pred)
    label_pred = np.where(y_score < threshold, np.zeros_like(label_pred), label_pred)
    # TN, FP, FN, TP
    tn, fp, fn, tp = metrics.confusion_matrix(y_true=y_label, y_pred=label_pred, labels=[0,1]).ravel()
    # Model Evaluation
    acc = metrics.accuracy_score(y_label, label_pred)
    mcc = metrics.matthews_corrcoef(y_label, label_pred)
    precision = metrics.precision_score(y_label, label_pred)
    recall = metrics.recall_score(y_label, label_pred)
    f1 = metrics.f1_score(y_label, label_pred)
    specificity = tn/(fp+tn)
    sensitivity = tp/(tp+fn)
    return fprs, tprs, thresholds_auc, pres, recs, np.append(thresholds_prc, [1], axis=0), tn, fp, fn, tp, acc, auc, mcc, precision, recall, specificity, sensitivity, f1, prauc, av_prc


