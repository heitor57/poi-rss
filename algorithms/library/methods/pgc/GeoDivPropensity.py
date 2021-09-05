import numpy as np
from sklearn.cluster import DBSCAN
# import os
# import sys
# module_path = os.path.abspath(os.path.join('..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)

import library.geo_utils as geo_utils
import collections
import scipy
from concurrent.futures import ProcessPoolExecutor

from parallel_util import run_parallel

import library.cat_utils as cat_utils
import library.geo_utils as geo_utils
import metrics
# (0.6118019290456039, 3.8000000000000003, 5), DBSCAN SILHOUETTE

class GeoDivPropensity():
    CHKS = 50 # chunk size for parallel pool executor
    _instance = None
    METHODS = ['walk'# ,'num_poi','num_cat','visits','walk_raw'
               # ,'ildg'
               # ,'inverse_walk'
               ,# 'dbscan','inv_dbscan','inv_wcluster','wcluster',
               # 'perfect'
    ]
    GEO_DIV_PROPENSITY_METHODS_PRETTY_NAME = {
        'walk': 'Mean radius of visited POIs',
        'num_poi': 'Number of visited POIs',
        'ildg': 'Geographical ILD',
        'inverse_walk': 'Inverse of mean radius of visited POIs',
        'dbscan': 'Using clusters of visited pois',
        'inv_dbscan': 'Inverse of dbscan',
        'inv_wcluster': 'Inverse weighted clustering',
        'wcluster': 'Weighted clustering',
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

        # import scipy.stats
        # print('mean walk',self.mean_walk)
        # print(scipy.stats.describe(self.users_mean_walk))
        import matplotlib.pyplot as plt
        num_bins = 50
        heights, bins, _ = plt.hist(self.users_mean_walk,bins=num_bins,color='k')
        bin_width = np.diff(bins)[0]
        bin_pos = bins[:-1] + bin_width / 2
        mask = (bin_pos <= self.mean_walk)
        fig, ax = plt.subplots(1,1)
        ax.bar(bin_pos[mask], heights[mask], width=bin_width, color='red')
        ax.bar(bin_pos[~mask], heights[~mask], width=bin_width, color='black')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        fig.savefig('resultadotemp.png',bbox_inches='tight')
        fig.savefig('resultadotemp.eps',bbox_inches='tight')
        raise SystemExit
        
        self.geo_div_method = geo_div_method
        self.GEO_METHODS = {
            "walk": self.geo_div_walk,
            "num_poi": self.geo_div_num_poi,
            "num_cat": self.geo_div_num_cat,
            "visits": self.geo_div_visits,
            "walk_raw": self.geo_div_walk_raw,
            "ildg": self.geo_div_ildg,
            "inverse_walk": self.geo_div_inverse_walk,
            "dbscan": self.geo_div_dbscan,
            "inv_dbscan": self.geo_div_inv_dbscan,
            "inv_wcluster": self.geo_div_inv_wcluster,
            "wcluster": self.geo_div_wcluster,
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
            md+=geo_utils.dist((lat,lon),(lats[i],longs[i]))
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
                md+=geo_utils.dist((lat,lon),(lats[i],longs[i]))
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
            md+=geo_utils.dist((lat,lon),(lats[i],longs[i]))
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
                md+=geo_utils.dist((lat,lon),(lats[i],longs[i]))
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

    @classmethod
    def geo_div_dbscan(cls, uid):
        self = cls.getInstance()
        km = 3.8
        min_samples = 5
        
        points = [self.poi_coos[lid]
                  for lid in np.nonzero(self.training_matrix[uid])[0]
                  # for _ in range(self.training_matrix[uid,lid])
        ]
        db = DBSCAN(eps=geo_utils.km_to_lat(km), min_samples=min_samples).fit(points)
        labels = db.labels_
        # Number of clusters in labels, ignoring noise if present.
        a = n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        b = n_noise_ = list(labels).count(-1)

        return ((b - a)/(max(a,b)) + 1)/2


    @classmethod
    def geo_div_inv_dbscan(cls, uid):
        self = cls.getInstance()
        km = 3.8
        min_samples = 5
        
        points = [self.poi_coos[lid]
                  for lid in np.nonzero(self.training_matrix[uid])[0]
                  # for _ in range(self.training_matrix[uid,lid])
        ]
        db = DBSCAN(eps=geo_utils.km_to_lat(km), min_samples=min_samples).fit(points)
        labels = db.labels_
        # Number of clusters in labels, ignoring noise if present.
        a = n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        b = n_noise_ = list(labels).count(-1)

        return 1-((b - a)/(max(a,b)) + 1)/2

    @classmethod
    def geo_div_inv_wcluster(cls, uid):
        self = cls.getInstance()
        km = 3.8
        min_samples = 5
        
        points = [self.poi_coos[lid]
                  for lid in np.nonzero(self.training_matrix[uid])[0]
                  # for _ in range(self.training_matrix[uid,lid])
        ]
        db = DBSCAN(eps=geo_utils.km_to_lat(km), min_samples=min_samples).fit(points)
        labels = db.labels_
        # Number of clusters in labels, ignoring noise if present.
        unique, counts = np.unique(labels, return_counts=True)
        u_c = dict(zip(unique, counts))
        wa = 0
        wb = 0
        for lab, amount in u_c.items():
            if lab != -1:
                wa += amount
            else:
                wb += amount
        a = n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        b = n_noise_ = list(labels).count(-1)
        a*= wa
        b*= wb

        return 1-((b - a)/(max(a,b)) + 1)/2


    @classmethod
    def geo_div_wcluster(cls, uid):
        self = cls.getInstance()
        km = 3.8
        min_samples = 5
        
        points = [self.poi_coos[lid]
                  for lid in np.nonzero(self.training_matrix[uid])[0]
                  # for _ in range(self.training_matrix[uid,lid])
        ]
        db = DBSCAN(eps=geo_utils.km_to_lat(km), min_samples=min_samples).fit(points)
        labels = db.labels_
        # Number of clusters in labels, ignoring noise if present.
        unique, counts = np.unique(labels, return_counts=True)
        u_c = dict(zip(unique, counts))
        wa = 0
        wb = 0
        for lab, amount in u_c.items():
            if lab != -1:
                wa += amount
            else:
                wb += amount
        a = n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        b = n_noise_ = list(labels).count(-1)
        a*= wa
        b*= wb

        return ((b - a)/(max(a,b)) + 1)/2

    def compute_div_propensity(self):
        func = self.GEO_METHODS.get(self.geo_div_method,
                                    lambda: "Invalid method")
        # self.geo_div_propensity = func()
        args=[(uid,) for uid in range(self.training_matrix.shape[0])]
        self.geo_div_propensity = run_parallel(func, args, self.CHKS)
        if self.geo_div_method == 'ildg':
            self.geo_div_propensity = self.geo_div_propensity/np.max(self.geo_div_propensity)
        self.geo_div_propensity = np.array(self.geo_div_propensity)

        # bins = np.append(np.arange(0,1,1/(3-1)),1)
        # centers = (bins[1:]+bins[:-1])/2
        # self.geo_div_propensity = bins[np.digitize(self.geo_div_propensity, centers)]

        # if self.geo_div_method == 'dbscan':
        #     self.geo_div_propensity[self.geo_div_propensity>=0.5] = 1
        #     self.geo_div_propensity[(self.geo_div_propensity<0.5) & (self.geo_div_propensity>=0.3)] = 0.5
        #     self.geo_div_propensity[self.geo_div_propensity<0.3] = 0
        return self.geo_div_propensity
