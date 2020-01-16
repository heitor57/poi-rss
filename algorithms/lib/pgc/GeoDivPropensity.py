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

class GeoDivPropensity():
    CHKS = 50 # chunk size for parallel pool executor
    _instance = None
    METHODS = ['walk','num_poi']

    @classmethod
    def getInstance(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance=cls(*args,**kwargs)
        return cls._instance

    def __init__(self,training_matrix,poi_coos,geo_div_method='walk'):
        self.training_matrix=training_matrix
        self.poi_coos=poi_coos

        self.mean_walk=self.cmean_dist_pois()
        self.users_mean_walk=self.cmean_dist_users()
        self.geo_div_method = geo_div_method
        self.GEO_METHODS = {
            "walk": self.geo_div_walk,
            "num_poi": self.geo_div_num_poi,
        }

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
    def geo_div_walk(self,uid):
        norm_prop=max((self.users_mean_walk[uid]/self.mean_walk[uid]),1)
        # self.geo_div_propensity=norm_prop
        return norm_prop


    @classmethod
    def geo_div_num_poi(cls, uid):
        self = cls.getInstance()
        lids = self.training_matrix[uid].nonzero()[0]
        return len(lids)/self.training_matrix.shape[1]


    def compute_geo_div_propensity(self):
        func = self.GEO_METHODS.get(self.geo_div_method,
                                    lambda: "Invalid method")
        # self.geo_div_propensity = func()
        args=[(uid,) for uid in range(self.training_matrix.shape[0])]
        self.geo_div_propensity = run_parallel(func, args, self.CHKS)
        return np.array(self.geo_div_propensity)
