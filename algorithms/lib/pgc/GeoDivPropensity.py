import numpy as np
# import os
# import sys
# module_path = os.path.abspath(os.path.join('..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)

import lib.geo_utils as geo_utils
import collections
import scipy
from concurrent.futures import ProcessPoolExecutor

from parallel_util import run_parallel

import lib.cat_utils as cat_utils
import metrics


class GeoDivPropensity():
    CHKS = 50 # chunk size for parallel pool executor
    _instance = None
    METHODS = ['walk','num_poi','num_cat','visits','walk_raw','ildg','inverse_walk']
    GEO_DIV_PROPENSITY_METHODS_PRETTY_NAME = {
        'walk': 'Mean radius of visited POIs',
        'num_poi': 'Number of visited POIs',
        'ildg': 'Geographical ILD',
        'inverse_walk': 'Inverse of mean radius of visited POIs'
    }

    @classmethod
    def getInstance(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance=cls(*args,**kwargs)
        elif len(args) > 0 or len(kwargs) > 0:
            cls._instance.__init__(*args,**kwargs)
        return cls._instance

    def __init__(self,training_matrix,poi_coos,poi_cats,undirected_category_tree,geo_div_method):
        self.training_matrix=training_matrix
        self.poi_coos=poi_coos
        self.poi_cats=poi_cats
        self.undirected_category_tree = undirected_category_tree
        self.users_categories_visits = cat_utils.get_users_cat_visits(self.training_matrix,
                                                                      self.poi_cats)
        self.mean_walk=self.cmean_dist_pois()
        self.users_mean_walk=self.cmean_dist_users()
        self.geo_div_method = geo_div_method
        self.GEO_METHODS = {
            "walk": self.geo_div_walk,
            "num_poi": self.geo_div_num_poi,
            "num_cat": self.geo_div_num_cat,
            "visits": self.geo_div_visits,
            "walk_raw": self.geo_div_walk_raw,
            "ildg": self.geo_div_ildg,
            "inverse_walk": self.geo_div_inverse_walk,
        }


        self.max_user_visits = self.training_matrix.sum(axis=1).max()
        
        self.geo_div_propensity=None


    def cmean_dist_pois(self):
        lats=np.array([])
        longs=np.array([])
        for i,j in self.poi_coos.items():
            lats=np.append(lats,j[0])
            longs=np.append(longs,j[1])
        lat,lon = np.mean(lats),np.mean(longs)
        md=0
        for i in range(len(lats)):
            md+=geo_utils.haversine(lat,lon,lats[i],longs[i])
        return md/len(lats)

    def cmean_dist_users(self):
        users_cmean=list()
        for i in range(self.training_matrix.shape[0]):
            lids=self.training_matrix[i].nonzero()[0]
            lats=np.array([])
            longs=np.array([])
            for lid in lids:
                lats=np.append(lats,self.poi_coos[lid][0])
                longs=np.append(longs,self.poi_coos[lid][1])
            lat,lon = np.mean(lats),np.mean(longs)
            md=0
            
            for i in range(len(lats)):
                md+=geo_utils.haversine(lat,lon,lats[i],longs[i])
            users_cmean.append(md/len(lats))
        return users_cmean

    def cmedian_dist_pois(self):
        lats=np.array([])
        longs=np.array([])
        for i,j in self.poi_coos.items():
            lats=np.append(lats,j[0])
            longs=np.append(longs,j[1])
        lat,lon = np.median(lats),np.median(longs)
        md=0
        for i in range(len(lats)):
            md+=geo_utils.haversine(lat,lon,lats[i],longs[i])
        return md/len(lats)

    def cmedian_dist_users(self):
        users_cmean=list()
        for i in range(self.training_matrix.shape[0]):
            lids=self.training_matrix[i].nonzero()[0]
            lats=np.array([])
            longs=np.array([])
            for lid in lids:
                lats=np.append(lats,self.poi_coos[lid][0])
                longs=np.append(longs,self.poi_coos[lid][1])
            lat,lon = np.median(lats),np.median(longs)
            md=0
            
            for i in range(len(lats)):
                md+=geo_utils.haversine(lat,lon,lats[i],longs[i])
            users_cmean.append(md/len(lats))
        return users_cmean

    @classmethod
    def geo_div_walk(cls,uid):
        self = cls.getInstance()
        norm_prop=min((self.users_mean_walk[uid]/self.mean_walk),1)
        # self.geo_div_propensity=norm_prop
        return norm_prop

    @classmethod
    def geo_div_walk_raw(cls,uid):
        self = cls.getInstance()
        norm_prop=self.mean_walk.copy()
        # self.geo_div_propensity=norm_prop
        return norm_prop

    @classmethod
    def geo_div_num_cat(cls, uid):
        self = cls.getInstance()
        cats_visits = self.users_categories_visits[uid]
        return len(cats_visits)/(len(self.undirected_category_tree)-1)

    @classmethod
    def geo_div_num_poi(cls, uid):
        self = cls.getInstance()
        lids = self.training_matrix[uid].nonzero()[0]
        return len(lids)/self.training_matrix.shape[1]

    @classmethod
    def geo_div_ildg(cls, uid):
        self = cls.getInstance()
        lids = self.training_matrix[uid].nonzero()[0]
        ildg = metrics.ildgk(lids,self.poi_coos)
        return ildg

    @classmethod
    def geo_div_visits(cls, uid):
        self = cls.getInstance()
        lids = self.training_matrix[uid].nonzero()[0]
        visits = self.training_matrix[uid,lids].sum()
        return visits/self.max_user_visits

    @classmethod
    def geo_div_inverse_walk(cls, uid):
        self = cls.getInstance()
        norm_prop=min((self.users_mean_walk[uid]/self.mean_walk),1)
        # self.geo_div_propensity=norm_prop
        return 1-norm_prop

    def compute_div_propensity(self):
        func = self.GEO_METHODS.get(self.geo_div_method,
                                    lambda: "Invalid method")
        # self.geo_div_propensity = func()
        args=[(uid,) for uid in range(self.training_matrix.shape[0])]
        self.geo_div_propensity = run_parallel(func, args, self.CHKS)
        if self.geo_div_method == 'ildg':
            self.geo_div_propensity = self.geo_div_propensity/np.max(self.geo_div_propensity)
        return np.array(self.geo_div_propensity)
