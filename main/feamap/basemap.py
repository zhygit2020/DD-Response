import sys
from pathlib import Path

from matplotlib.colors import Colormap
prj_path = Path(__file__).parent.resolve().parent.resolve()
sys.path.append(str(prj_path))

from feamap.utils.logtools import print_info, print_warn, print_error
from feamap.utils.matrixopt import Scatter2Grid, Scatter2Array 

from feamap.utils import vismap
from feamap.config import load_config
from feamap.preparation import fgp_from_smiles, des_from_smiles, geneprofile_from_local

from sklearn.manifold import TSNE, MDS
from sklearn.utils import shuffle
from joblib import Parallel, delayed, load, dump
from umap import UMAP
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from copy import copy

class Base:
    
    def __init__(self):
        pass
        
    def _save(self, filename):
        return dump(self, filename)
        
    def _load(self, filename):
        return load(filename)
 

    def MinMaxScaleClip(self, x, xmin, xmax):
        scaled = (x - xmin) / ((xmax - xmin) + 1e-8)
        return scaled.clip(0, 1)

    def StandardScaler(self, x, xmean, xstd):
        return (x-xmean) / (xstd + 1e-8) 
    
    
class Map(Base):
    
    def __init__(self, disttype = 'ALL', datatype = 'drug_fea', ftype = 'fingerprint', flist = [], fmap_type = 'grid', fmap_shape = None, split_channels = True, metric = 'cosine', var_thr = 1e-4, ):
    
        super().__init__()
        assert ftype in ['descriptor', 'fingerprint', 'geneprofile'], 'no such feature type supported!'        
        assert fmap_type in ['scatter', 'grid'], 'no such feature map type supported!'
       
        self.datatype = datatype
        self.ftype = ftype
        self.metric = metric
        self.method = None
        self.isfit = False

        if ftype == 'fingerprint':
            _, self.bitsinfo, colormaps = fgp_from_smiles(pd.DataFrame({'drugid':'test','cid':'x','drug_name':'default','drugid':'DRx','isosmiles':'CC'}, index=[0]))
        elif ftype == 'descriptor':
            _, self.bitsinfo, colormaps = des_from_smiles(pd.DataFrame({'drugid':'test','cid':'x','drug_name':'default','drugid':'DRx','isosmiles':'CC'}, index=[0]))
        elif ftype == 'geneprofile':
            celllines = pd.read_csv(prj_path / 'data' / 'original_data' / 'all-data-merge-cell.csv', index_col=0, low_memory=False)
            _, self.bitsinfo, colormaps = geneprofile_from_local(celllines.iloc[0:1,:])
        
        # default we will load the precomputed matrix
        dist_matrix = load_config(disttype, datatype, ftype, metric)
        feature_order = dist_matrix.index.tolist()
        feat_seq_dict = dict(zip(feature_order, range(len(feature_order))))
        scale_info = load_config(disttype, datatype, ftype, 'scale')
        scale_info['min']=scale_info['min'].astype(float)
        scale_info['max']=scale_info['max'].astype(float)
        scale_info = scale_info[scale_info['var'] > var_thr]
        slist = scale_info.index.tolist()

        if flist == []:
            self.channel_col = list(colormaps.keys())
            self.channel_col.remove("NaN")
            flist = list(dist_matrix.columns)
        else:
            self.channel_col = copy(flist)
            flist = self.bitsinfo.loc[~(self.bitsinfo[flist].sum(axis=1)==0)].IDs.tolist()
        
        #fix input feature's order as random order
        final_list = list(set(slist) & set(flist))
        final_list.sort(key = lambda x:feat_seq_dict.get(x))

        dist_matrix = dist_matrix.loc[final_list][final_list]
        
        self.dist_matrix = dist_matrix
        self.flist = final_list
        self.scale_info = scale_info.loc[final_list]
        
        self.colormaps = colormaps
        self.fmap_type = fmap_type
        
        if fmap_type == 'grid':
            S = Scatter2Grid()
        else:
            if fmap_shape == None:
                N = len(self.flist)
                l = np.int(np.sqrt(N))*2
                fmap_shape = (l, l)                
            S = Scatter2Array(fmap_shape)
        
        self._S = S
        self.split_channels = split_channels        


    def _fit_embedding(self, 
                        method = 'tsne',  
                        n_components = 2,
                        random_state = 1,  
                        verbose = 2,
                        n_neighbors = 30,
                        min_dist = 0.1,
                        **kwargs):
        
        """
        parameters
        -----------------
        method: {'tsne', 'umap', 'mds'}, algorithm to embedd high-D to 2D
        kwargs: the extra parameters for the conresponding algorithm
        """
        dist_matrix = self.dist_matrix
        if 'metric' in kwargs.keys():
            metric = kwargs.get('metric')
            kwargs.pop('metric')
            
        else:
            metric = 'precomputed'

        if method == 'tsne':
            embedded = TSNE(n_components=n_components, 
                            random_state=random_state,
                            metric = metric,
                            verbose = verbose,
                            **kwargs)
        elif method == 'umap':
            embedded = UMAP(n_components = n_components, 
                            n_neighbors = n_neighbors,
                            min_dist = min_dist,
                            verbose = verbose,
                            random_state=random_state, 
                            metric = metric, **kwargs)
            
        elif method =='mds':
            if 'metric' in kwargs.keys():
                kwargs.pop('metric')
            if 'dissimilarity' in kwargs.keys():
                dissimilarity = kwargs.get('dissimilarity')
                kwargs.pop('dissimilarity')
            else:
                dissimilarity = 'precomputed'
                
            embedded = MDS(metric = True, 
                           n_components= n_components,
                           verbose = verbose,
                           dissimilarity = dissimilarity, 
                           random_state = random_state, **kwargs)
        
        embedded = embedded.fit(dist_matrix)    

        df = pd.DataFrame(embedded.embedding_, index = self.flist,columns=['x', 'y'])
        typemap = self.bitsinfo.set_index('IDs')
        # print(typemap)
        df = df.join(typemap)
        # print(df)
        # df['Channels'] = df['Subtypes'] ########################
        self.df_embedding = df
        self.embedded = embedded


    def fit(self, 
            method = 'umap', min_dist = 0.1, n_neighbors = 30,
            verbose = 2, random_state = 1, **kwargs): 
        """
        parameters
        -----------------
        method: {'tsne', 'umap', 'mds'}, algorithm to embedd high-D to 2D
        kwargs: the extra parameters for the conresponding method
        """
        if 'n_components' in kwargs.keys():
            kwargs.pop('n_components')
            
        ## embedding  into a 2d 
        assert method in ['tsne', 'umap', 'mds'], 'no support such method!'
        
        self.method = method
        
        ## 2d embedding first
        self._fit_embedding(method = method,
                            n_neighbors = n_neighbors,
                            random_state = random_state,
                            min_dist = min_dist, 
                            verbose = verbose,
                            n_components = 2, **kwargs)

        
        if self.fmap_type == 'scatter':
            ## naive scatter algorithm
            print_info('Applying naive scatter feature map...')
            self._S.fit(self.df_embedding, self.split_channels, channel_col = self.channel_col)
            print_info('Finished')
            
        else:
            ## linear assignment algorithm 
            print_info('Applying grid feature map(assignment), this may take several minutes(1~30 min)')
            self._S.fit(self.df_embedding, self.split_channels, channel_col = self.channel_col)
            print_info('Finished')
        
        ## fit flag
        self.isfit = True
        self.fmap_shape = self._S.fmap_shape


    def transform(self, arr, scale = True, scale_method = 'minmax',):
        
        if not self.isfit:
            print_error('please fit first!')
            return

        # if (scale) & (self.ftype == 'descriptor'):
        if scale:
            if scale_method == 'standard':
                df = self.StandardScaler(arr, self.scale_info['mean'], self.scale_info['std']).to_frame().T
            elif scale_method == 'minmax':
                df = self.MinMaxScaleClip(arr, self.scale_info['min'], self.scale_info['max']).to_frame().T
            elif scale_method == 'None':
                df = arr.to_frame().T

        df = df[self.flist]

        vector_1d = df.values[0]

        fmap = self._S.transform(vector_1d)
        return np.nan_to_num(fmap)   

        
    def batch_transform(self, 
                        arrs, 
                        scale = True, 
                        scale_method = 'minmax',
                        n_jobs=4):
    
        """
        parameters
        --------------------
        smiles_list: list of smiles strings
        scale: bool, if True, we will apply MinMax scaling by the precomputed values
        scale_method: {'minmax', 'standard'}
        n_jobs: number of parallel
        """
        
        print("map batch_transform")

        x = []
        ids = []
        for id, arr in tqdm(arrs.iterrows()):
            res = self.transform(arr, scale, scale_method)
            x.append(res)
            ids.append(id)
        
        X = np.stack(x, axis=0)
        return X, ids


    def rearrangement(self, orignal_X, target_mp):

        """
        Re-Arragement feature maps X from orignal_mp's to target_mp's style, in case that feature already extracted but the position need to be refit and rearrangement.

        parameters
        -------------------
        orignal_X: the feature values transformed from orignal_mp(object self)
        target_mp: the target feature map object

        return
        -------------
        target_X, shape is (N, W, H, C)
        """
        assert self.flist == target_mp.flist, print_error('Input features list is different, can not re-arrangement, check your flist by mp.flist method' )
        assert len(orignal_X.shape) == 4, print_error('Input X has error shape, please reshape to (samples, w, h, channels)')
        
        idx = self._S.df.sort_values('indices').idx.tolist()
        idx = np.argsort(idx)

        N = len(orignal_X) #number of sample
        M = len(self.flist) # number of features
        res = []
        for i in tqdm(range(N), ascii=True):
            x = orignal_X[i].sum(axis=-1)
            vector_1d_ordered = x.reshape(-1,)
            vector_1d_ordered = vector_1d_ordered[:M]
            vector_1d = vector_1d_ordered[idx]
            fmap = target_mp._S.transform(vector_1d)
            res.append(fmap)
        return np.stack(res)


    def plot_scatter(self, htmlpath='./', htmlname=None, radius = 3):
        """radius: the size of the scatter, must be int"""
        df_scatter, H_scatter = vismap.plot_scatter(self,  
                                htmlpath=htmlpath, 
                                htmlname=htmlname,
                                radius = radius)
        
        self.df_scatter = df_scatter
        return H_scatter   
        

    def plot_grid(self, htmlpath='./', htmlname=None):
        
        if self.fmap_type != 'grid':
            return
        
        df_grid, H_grid = vismap.plot_grid(self,  
                                htmlpath=htmlpath, 
                                htmlname=htmlname)
        
        self.df_grid = df_grid
        return H_grid       


    def load(self, filename):
        mp = self._load(filename)
        print('map loaded info: ', mp.ftype, '_', len(mp.flist), '_', mp.metric, '_', mp.method)
        return mp


    def save(self, filename):
        return self._save(filename)
