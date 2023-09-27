from bisect import bisect
from typing import DefaultDict
from rdkit import DataStructs
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import numpy as np
import pandas as pd
from pathlib import Path
from sqlalchemy import column
import tqdm
import scipy.io as scio
import sys
import sklearn
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from joblib import Parallel, delayed
from collections import defaultdict
from collections import OrderedDict
import seaborn as sns
import math
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import ClusterCentroids
from imblearn.over_sampling import RandomOverSampler

from feamap.utils import distances, calculator
from feamap.feature.fingerprint import GetAtomPairFPs, GetAvalonFPs, GetRDkitFPs, GetMorganFPs, GetEstateFPs, GetMACCSFPs, GetPharmacoErGFPs, GetPharmacoPFPs, GetPubChemFPs, GetTorsionFPs, GetMHFP6, GetMAP4
from feamap.feature.descriptor import GetAutocorr, _AutocorrNames, GetCharge, _ChargeNames, GetConnectivity, _ConnectivityNames, GetConstitution, _ConstitutionNames, GetEstate, _EstateNames, GetFragment, _FragmentNames, GetKappa, _KappaNames, GetMOE, _MOENames, GetPath, _PathNames, GetProperty, _PropertyNames, GetTopology, _TopologyNames, GetMatrix, _MatrixNames, GetInfoContent, _InfoContentNames
from feamap import summary

prj_path = Path(__file__).parent.resolve().parent.resolve()

def pair_statistic(label, type):
    drugs = label['drugid'].value_counts(sort=False)
    drugs.name = 'pair_counts'
    drugs = drugs.to_frame().assign(gdsc_counts = np.nan, ccle_counts = np.nan, ctrp_counts = np.nan, pos_counts = np.nan, neg_counts = np.nan)
    for drug in drugs.index:
        x = label.loc[label['drugid']==drug]
        source_counts = x['source'].value_counts(sort=False).to_dict()
        posneg_counts = x['label'].value_counts(sort=False).to_dict()
        drugs.at[drug,'gdsc_counts'] = source_counts.get('GDSC')
        drugs.at[drug,'ccle_counts'] = source_counts.get('CCLE')
        drugs.at[drug,'ctrp_counts'] = source_counts.get('CTRP')
        drugs.at[drug,'pos_counts'] = posneg_counts.get(1)
        drugs.at[drug,'neg_counts'] = posneg_counts.get(0)
    drugs.to_csv(prj_path / 'data' / f'original_data' / f'drugpair_statistic_{type}.csv')

    celllines = label['gdsc-id'].value_counts(sort=False)
    celllines.name = 'pair_counts'
    celllines = celllines.to_frame().assign(gdsc_counts = np.nan, ccle_counts = np.nan, ctrp_counts = np.nan, pos_counts = np.nan, neg_counts = np.nan)
    for cellline in celllines.index:
        x = label.loc[label['gdsc-id']==cellline]
        source_counts = x['source'].value_counts(sort=False).to_dict()
        posneg_counts = x['label'].value_counts(sort=False).to_dict()
        celllines.at[cellline,'gdsc_counts'] = source_counts.get('GDSC')
        celllines.at[cellline,'ccle_counts'] = source_counts.get('CCLE')
        celllines.at[cellline,'ctrp_counts'] = source_counts.get('CTRP')
        celllines.at[cellline,'pos_counts'] = posneg_counts.get(1)
        celllines.at[cellline,'neg_counts'] = posneg_counts.get(0)
    celllines.to_csv(prj_path / 'data' / f'original_data' / f'celllinepair_statistic_{type}.csv')

def data_washing(params, label):
    drugpair_statistic = pd.read_csv(prj_path / 'data' / 'original_data' / 'drugpair_statistic_original.csv', index_col=0)
    celllinepair_statistic = pd.read_csv(prj_path / 'data' / 'original_data' / 'celllinepair_statistic_original.csv', index_col=0)

    print(f"sample num before washing: {label.shape[0]}")
    drugs_ = drugpair_statistic.loc[(drugpair_statistic['pos_counts']<5) | (drugpair_statistic['neg_counts']<5)]
    celllines_ = celllinepair_statistic.loc[(celllinepair_statistic['pos_counts']<5) | (celllinepair_statistic['neg_counts']<5)]

    print(f"len(label): {len(label)}")
    sample_ = pd.concat([drugs_,celllines_])
    label = label.loc[~((label['drugid'].isin(sample_.index))|(label['gdsc-id'].isin(sample_.index)))]
    print(f"len(label): {len(label)}")

    # randomly test setting
    x_trainval, x_test, y_trainval, y_test = train_test_split(label, label['label'], test_size=0.1, random_state=params.random_seed, shuffle=True, stratify=label['label'])
    print(f"len(x_test),len(x_trainval): {len(x_test),len(x_trainval)}")
    savecv_path = prj_path / 'data' / 'processed_data' / 'split_cvdata'
    savecv_path.mkdir(parents=True, exist_ok=True)
    pair_statistic(x_trainval,'washedtrainval')
    pair_statistic(x_test,'washedtest')

    return x_trainval, x_test

def drug_data():
    drugs = pd.read_csv(prj_path / 'data' / 'original_data' / 'all-data-merge-drug.csv', index_col=None, low_memory=False)
    # drugs.columns = ['drugid', 'cid']
    num_drugs = len(drugs.index)
    # print(drugs)
    # #        drugid         cid
    # # 0      DR00499      60700
    return drugs, num_drugs

def cellline_data():
    celllines = pd.read_csv(prj_path / 'data' / 'original_data' / 'all-data-merge-cell.csv', index_col=0, low_memory=False)
    # celllines.columns = ['gdsc-id', 'model_name']
    num_celllines = len(celllines.index)
    # print(num_celllines) # 1123
    return celllines, num_celllines

def relationship_data():
    label = pd.read_csv(prj_path / 'data' / 'original_data' / 'cell-drug-index-source.csv', low_memory=False)
    pair_statistic(label, 'original')
    return label

def split_data(params, label, ):
    print("WASHING DATA")
    _x_trainval, _x_test = data_washing(params, label)
    print(f"Split train, valid and test datasets")
    sources = label['source'].drop_duplicates()
    cv_data = defaultdict(list)
    for s in sources:
        x_trainval = _x_trainval.loc[_x_trainval['source']==s]
        y_trainval = x_trainval['label']
        print('np.count_nonzero(y_trainval)/len(y_trainval)', s, ": ", np.count_nonzero(y_trainval)/len(y_trainval))
        cv_data_s = defaultdict(list)
        skf = StratifiedKFold(n_splits=params.kfold_num, shuffle=True, random_state = params.random_seed)
        for k, (t,v) in enumerate(skf.split(x_trainval, y_trainval)):
            cv_data_s[k]=(x_trainval.iloc[t],x_trainval.iloc[v])

        cv_data[s]=cv_data_s

    for k in range(params.kfold_num):
        train_k = shuffle(pd.concat([cv_data[s][k][0] for s in sources]), random_state = params.random_seed)
        valid_k = shuffle(pd.concat([cv_data[s][k][1] for s in sources]), random_state = params.random_seed)
        # test_k = shuffle(pd.concat([cv_data[s][k][2] for s in sources]), random_state = params.random_seed)
        test_k = shuffle(_x_test, random_state = params.random_seed)
        savecv_path = prj_path / 'data' / 'processed_data' / 'split_cvdata' / f'{k}th_fold'
        savecv_path.mkdir(parents=True, exist_ok=True)
        train_k.to_csv(savecv_path / 'train_k.csv' )
        valid_k.to_csv(savecv_path / 'valid_k.csv' )
        test_k.to_csv(savecv_path / 'test_k.csv' )

def des_from_smiles(smiles, feature_dict=None): # feature_dict: dict parameters for the corresponding fingerprint type, say: {'Property':['MolWeight', 'MolSLogP']}
    mapfunc = {GetProperty:'Property', 
            GetConstitution:'Constitution', 
            #structure
            GetAutocorr:'Autocorr',
            GetFragment: 'Fragment',
            #state
            GetCharge:'Charge',
            GetEstate:'Estate',
            GetMOE:'MOE',
            ## graph
            GetConnectivity:'Connectivity', 
            GetTopology:'Topology', 
            GetKappa:'Kappa', 
            GetPath:'Path', 
            GetMatrix:'Matrix', 
            GetInfoContent: 'InfoContent'}
    mapkey = dict(map(reversed, mapfunc.items()))
    _subclass_ = {'Property':_PropertyNames, 
                'Constitution':_ConstitutionNames,
                'Autocorr':_AutocorrNames,
                'Fragment':_FragmentNames,
                'Charge':_ChargeNames,
                'Estate':_EstateNames,
                'MOE':_MOENames,
                'Connectivity':_ConnectivityNames,
                'Topology':_TopologyNames,
                'Kappa':_KappaNames,
                'Path':_PathNames,
                'Matrix':_MatrixNames,
                'InfoContent':_InfoContentNames}
    colormaps = {'Property': '#ff6a00',
                'Constitution': '#ffd500',
                'Autocorr': '#bfff00',
                'Connectivity': '#4fff00',
                'Topology': '#00ff1b',
                'Kappa': '#00ff86', 
                'Path': '#00fff6',             
                'Fragment': '#009eff',
                'Charge': '#0033ff',
                'Estate': '#6568f7',  # #3700ff
                'MOE': '#a700ff',
                'Matrix': '#ff00ed',
                'InfoContent': '#ff0082',
                'NaN': '#000000'}
    if not feature_dict:
        factory = mapkey
        feature_dict = _subclass_
        flag = 'all'
        cm = colormaps
    else:
        factory = {key:mapkey[key] for key in set(feature_dict.keys()) & set(mapkey)}
        feature_dict = feature_dict
        flag = 'auto'
    assert factory != {}, 'types of feature %s can be used' % list(mapkey.keys())
    keys = []
    for key, lst in feature_dict.items():
        if not lst:
            nlst = _subclass_.get(key)
        else:
            nlst = lst
        keys.extend([(v, key) for v in nlst])
    bitsinfo = pd.DataFrame(keys, columns=['IDs', 'Subtypes'])
    
    des_info = pd.DataFrame([])
    for idx, i in bitsinfo['IDs'].items():
        des_info[i] = []
    des_info['index']=smiles.drugid
    des_info.set_index(['index'], inplace=True)

    with tqdm.tqdm(smiles.iterrows()) as tq:
        print('...working on desciptor generating...')
        for step, (i, row) in enumerate(tq):
            try:
                mol = Chem.MolFromSmiles(row['isosmiles'])
                _all = OrderedDict()
                for key,func in factory.items():
                    flist = feature_dict.get(key)
                    dict_res = func(mol)
                    if (flag == 'all') | (not flist):
                        _all.update(dict_res)
                    else:
                        for k in flist:
                            _all.update({k:dict_res.get(k)})

                arr = np.fromiter(_all.values(), dtype=float)
                arr[np.isinf(arr)] = np.nan #convert inf value with nan
            except:
                arr = np.nan * np.ones(shape=(len(bitsinfo['IDs']), ), dtype=float)
                print(f"error when calculating {row['drug_name']}, please check out")
            
            des_info.loc[row['drugid']] = arr
            tq.set_postfix({'working on the drug of': row['drug_name']}, refresh=False)

    bitsinfo = bitsinfo.join(pd.get_dummies(bitsinfo.Subtypes))
    return des_info, bitsinfo, colormaps

def fgp_from_smiles(smiles, feature_dict=None): # feature_dict: dict parameters for the corresponding fingerprint type, say: {'AtomPairFP':{'nBits':2048}}
    mapfunc = {GetMorganFPs:'MorganFP', GetRDkitFPs: 'RDkitFP', GetAtomPairFPs:'AtomPairFP', GetTorsionFPs:'TorsionFP', GetAvalonFPs:'AvalonFP', GetEstateFPs:'EstateFP', GetMACCSFPs:'MACCSFP', GetPharmacoErGFPs:'PharmacoErGFP', GetPharmacoPFPs: 'PharmacoPFP', GetPubChemFPs:'PubChemFP', GetMHFP6:'MHFP6', GetMAP4:'MAP4',}
    mapkey = dict(map(reversed, mapfunc.items()))
    colors = sns.palettes.color_palette('Paired', n_colors=len(mapkey)).as_hex()
    fps = {'MorganFP':{},'RDkitFP':{}, 'AtomPairFP':{},'TorsionFP':{},'AvalonFP':{},'EstateFP':{},'PubChemFP':{},'PharmacoErGFP':{},'PharmacoPFP':{},'MHFP6':{}, 'MAP4':{},'MACCSFP':{},}
    colormaps = dict(zip(fps, colors)) 
    colormaps.update({'NaN': '#000000'})
    if not feature_dict:
        factory = mapkey
        flag = 'all'
        feature_dict = fps
        cm = colormaps
    else:
        keys = [key for key in set(feature_dict.keys()) & set(mapkey)]
        flag = 'auto'
        factory = {}
        cm = {}
        for k, v in mapkey.items():
            if k in keys:
                factory[k] = mapkey[k]
                cm[k] = colormaps[k]
    assert factory != {}, 'types of feature %s can be used' % list(mapkey.keys())
    _length, keys = [], []
    for key,func in factory.items():
        kwargs = feature_dict.get(key)
        if type(kwargs) == dict:
            _ = func(Chem.MolFromSmiles('CC', **kwargs))
        else:
            _ = func(Chem.MolFromSmiles('CC'))
        _length.append(len(_))
    for key, length in zip(factory.keys(),  _length):
        keys.extend([(key+str(i), key) for i in range(length)])
    bitsinfo = pd.DataFrame(keys, columns=['IDs', 'Subtypes'])
    
    fgp_info = pd.DataFrame([])
    for idx, i in bitsinfo['IDs'].items():
        fgp_info[i] = []
    fgp_info['index']=smiles.drugid
    fgp_info.set_index(['index'], inplace=True)

    with tqdm.tqdm(smiles.iterrows()) as tq:
        print('...working on fingerprint generating...')
        for step, (i, row) in enumerate(tq):
            try:
                mol = Chem.MolFromSmiles(row['isosmiles'])
                _all = []
                for key,func in factory.items():
                    kwargs = feature_dict.get(key)
                    if type(kwargs) == dict:
                        arr = func(mol, **kwargs)
                    else:
                        arr = func(mol)
                    _all.append(arr)
                concantefp = np.concatenate(_all).astype(float)
            except:
                concantefp = np.zeros(shape=(len(bitsinfo['IDs']), ), dtype=float)
                print(f"error when calculating {row['drug_name']}, please check out")
            
            fgp_info.loc[row['drugid']] = concantefp
            tq.set_postfix({'working on the drug of': row['drug_name']}, refresh=False)

    bitsinfo = bitsinfo.join(pd.get_dummies(bitsinfo.Subtypes))
    return fgp_info, bitsinfo, cm

def geneprofile_from_local(cellline, feature_dict=None):
    class_factory = ['Cellular_Processes', 'Developmental_Biology', 'Drug_Development', 'Environmental_Information_Processing', 'Genetic_Information_Processing', 'Human_Diseases', 'Metabolism', 'Organismal_Systems']
    bitsinfo = pd.read_csv(prj_path / 'data' / 'original_data' / 'reactome-kegg-class-width.csv',index_col=None, header=0, low_memory=False)

    _df = bitsinfo.melt('IDs')
    _ds = _df.loc[_df['value'].eq(1)].groupby(by='IDs').apply(lambda x:','.join(x['variable']))
    _ds.name = 'Subtypes'
    bitsinfo=bitsinfo.join(_ds, on='IDs')

    colors = ['#FF6633','#9900CC','#0033FF','#339900','#666600','#FFCC33','#FF0000','#00FFFF']
    colors = sns.palettes.color_palette("PuBu_d", n_colors=len(class_factory)).as_hex()
    colormaps = dict(zip(class_factory, colors)) 
    colormaps.update({'NaN': '#000000'})

    if not feature_dict:
        flag = 'all'
        keys = class_factory
        cm = colormaps
    else:
        keys = [key for key in set(feature_dict.keys()) & set(class_factory)]
        flag = 'auto'
        cm = {}
        for k in class_factory:
            if k in keys:
                cm[k] = colormaps[k]

    cel_info = cellline[bitsinfo['IDs']]

    return cel_info, bitsinfo, cm

def to_dist_matrix(data, datatype, idx, tag, methods = ['correlation', 'cosine', 'jaccard']):
    df_dic = {}
    for method in methods:
        res = calculator.pairwise_distance(data, n_cpus=12, method=method)
        res = np.nan_to_num(res,copy=False)
        df = pd.DataFrame(res,index=idx,columns=idx)
        save_path = prj_path / 'data' / 'processed_data' / f'{datatype}' / 'scale'
        save_path.mkdir(parents=True, exist_ok=True)
        df.to_pickle(save_path / f'{tag}_{method}.cfg')
        df_dic[method] = df
    return df_dic

def drug_fea(drugs, type='fingerprint'):
    if type=='fingerprint':
        print('...drug feature on fingerprint...')
        drugs_fea, bitsinfo, colormaps_d = fgp_from_smiles(drugs, feature_dict={'MACCSFP':{},'PharmacoErGFP':{},'PubChemFP':{}})
    elif type=='descriptor':
        print('...drug feature on descriptor...')
        drugs_fea, bitsinfo, colormaps_d = des_from_smiles(drugs, feature_dict=None)
    
    print('drugs_fea.shape before integrated: ',drugs_fea.shape)
    return drugs_fea, bitsinfo

def cellline_fea(celllines):
    print('...loading cellline-gene profiles...')
    cells_fea, bitsinfo, colormaps_c = geneprofile_from_local(celllines, feature_dict=None)
    print('cells_fea.shape before integrated: ',cells_fea.shape)
    return cells_fea, bitsinfo


def fea_statistic(fea):
    S = summary.Summary(n_jobs = 10)
    res= []
    for i in tqdm.tqdm(range(fea.shape[1])):
        r = S._statistics_one(fea.values, i)
        res.append(r)
        
    df = pd.DataFrame(res)
    df.index = fea.columns
    return df

def factor_int(n):
    val = math.ceil(math.sqrt(n))
    val2 = int(n/val)
    while val2 * val < float(n):
        val2 += 1
    return val, val2