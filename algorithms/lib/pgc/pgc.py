import numpy as np
# import os
# import sys
# module_path = os.path.abspath(os.path.join('..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)

import lib.geo_utils as geo_utils
import collections
import scipy
import lib.geocat.objfunc as geocat
import lib.cat_utils as cat_utils
from concurrent.futures import ProcessPoolExecutor

class PgcRunner():
    def __init__(self,training_matrix,poi_coos,users_categories_visits,undirected_category_tree,geo_method,cat_method):
        self.training_matrix=training_matrix
        self.poi_coos=poi_coos
        self.users_categories_visits=users_categories_visits
        self.undirected_category_tree=undirected_category_tree
        self.geo_method=geo_method
        self.cat_method=cat_method
    
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


    def geo_div_propensity(self,users_cmean_dist,cmean_dist_city):
        norm_prop=(users_cmean_dist/cmean_dist_city)
        norm_prop[norm_prop>1]=1
        return norm_prop

    def cat_div_std_norm(self,cats_visits):
        std=np.std(cats_visits)
        if std == 0:
            return 0
        else:
            return 1-2*std/(np.max(cats_visits)-np.min(cats_visits))
    
    def cat_div_mad_norm(self,cats_visits):
        std=scipy.stats.median_absolute_deviation(cats_visits)
        if std == 0:
            return 0
        else:
            return 1-std/(np.max(cats_visits)-np.min(cats_visits))
    

    def cat_div_skew(self,cats_visits):
        
        pass


    def cat_div_ld(self,cats_visits):
        dis_sum=0
        for cat1 in cats_visits.keys():
            for cat2 in cats_visits.keys():
                if cat1 != cat2:
                    dis_sum+=geocat.category_dis_sim(cat1,cat2,self.undirected_category_tree)
        l=len(cats_visits)
        return dis_sum/(l**2-l)          

    def cat_div_propensity(self,method="cat_div_std_norm"):
        switcher = {
            "cat_div_std_norm": self.cat_div_std_norm,
            "cat_div_skew": self.cat_div_skew,
            "cat_div_mad_norm":self.cat_div_mad_norm,
            "cat_div_ld":self.cat_div_ld,
        }
        func = switcher.get(method, lambda: "Invalid method")
        
        executor = ProcessPoolExecutor()
        futures=[]
        for i,cat_visits in enumerate(self.users_categories_visits):
            futures.append(executor.submit(func,cat_visits))
        results = [future.result() for future in futures]
        return results



# def tcat_div_propensity(training_matrix,poi_cats):
#     dis_sum=0
#     for uid in range(training_matrix.shape[0]):
#         pass

