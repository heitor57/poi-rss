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

class GeoDivPropensity():
    def __init__(self,training_matrix,poi_coos):
        self.training_matrix=training_matrix
        self.poi_coos=poi_coos

        self.mean_walk=self.cmean_dist_pois()
        self.user_mean_walk=self.cmean_dist_users()

        self.GEO_METHODS = {
            "walk": self.geo_div_walk,
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

    def geo_div_walk(self):
        norm_prop=(self.user_mean_walk/self.mean_walk)
        norm_prop[norm_prop>1]=1
        self.geo_div_propensity=norm_prop
        return norm_prop
    
