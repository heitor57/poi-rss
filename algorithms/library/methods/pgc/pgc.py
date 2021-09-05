import numpy as np
# import os
# import sys
# module_path = os.path.abspath(os.path.join('..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)

import library.geo_utils as geo_utils
import collections
import scipy
import library.geocat.objfunc as geocat
import library.cat_utils as cat_utils
from concurrent.futures import ProcessPoolExecutor

class PgcRunner():
    def __init__(self,training_matrix,poi_coos,users_categories_visits,undirected_category_tree,cat_div_method='ld'):
        self.training_matrix=training_matrix
        self.poi_coos=poi_coos
        self.users_categories_visits=users_categories_visits
        self.undirected_category_tree=undirected_category_tree
        self.cat_div_method=cat_div_method

        self.mean_walk=self.cmean_dist_pois()
        self.user_mean_walk=self.cmean_dist_users()

        self.GEO_METHODS = {
            "walk": self.geo_div_walk,
        }
        self.CAT_METHODS = {
            "std_norm": self.cat_div_std_norm,
            "mad_norm": self.cat_div_mad_norm,
            "ld": self.cat_div_ld
        }

        self.geo_div_propensity=None
        self.cat_div_propensity=None

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

    def compute_cat_div_propensity(self):
        # switcher = {
        #     "cat_div_std_norm": self.cat_div_std_norm,
        #     "cat_div_skew": self.cat_div_skew,
        #     "cat_div_mad_norm":self.cat_div_mad_norm,
        #     "cat_div_ld":self.cat_div_ld,
        # }
        func = self.CAT_METHODS.get(self.cat_div_method, lambda: "Invalid method")
        
        executor = ProcessPoolExecutor()
        futures=[]
        for i,cat_visits in enumerate(self.users_categories_visits):
            futures.append(executor.submit(func,cat_visits))
        self.cat_div_propensity = [future.result() for future in futures]
        return self.cat_div_propensity



# def tcat_div_propensity(training_matrix,poi_cats):
#     dis_sum=0
#     for uid in range(training_matrix.shape[0]):
#         pass

