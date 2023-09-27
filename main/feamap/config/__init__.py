from feamap.utils.logtools import print_info
import sys
import gdown
import pandas as pd
import os
from pathlib import Path

prj_path = Path(__file__).parent.resolve().parent.resolve().parent.resolve()

def load_config(disttype = 'ALL', datatype = 'drug_fea', ftype = 'descriptor', metric = 'cosine'):
    if metric=='scale':
        if datatype == 'drug_fea':
            df = pd.read_pickle(prj_path / 'feamap' / 'config' / f'trans_from_{disttype}' / f'{ftype}_scale.cfg')
        elif datatype == 'cellline_fea':
            df = pd.read_pickle(prj_path / 'feamap' / 'config' / f'trans_from_{disttype}' / f'geneprofile_scale.cfg')
    else:
        try:
            df = pd.read_pickle(prj_path / 'feamap' / 'config' / f'trans_from_{disttype}' / f'{ftype}_{metric}.cfg')
        except:
            print('Error while loading feature distance matrix')
            sys.exit()
    return df
