from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy
from tqdm import tqdm
from parallel_util import run_parallel

import cat_utils
import objfunc as geocat
from Binomial import Binomial
import metrics
import cat_utils




class CatDivPropensity():
    CHKS = 50 # chunk size for parallel pool executor
    _instance = None
    METHODS = [# 'std_norm','ild','raw_std',
               # 'num_cat',
        'inv_num_cat',
               # 'binomial','poi_ild'
    ]
    CAT_DIV_PROPENSITY_METHODS_PRETTY_NAME = {
        'poi_ild': 'ILD',
        'num_cat': 'Number of categories visited',
        'binomial': 'Binomial',
        'std_norm': r'$\sigma$(STD) of categories visits',
        'raw_std': '1-std_norm',
        'ild': 'Visited categories ILD',
        'inv_num_cat': 'Inverse number of categories visited',
    }
    @classmethod
    def getInstance(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance=cls(*args,**kwargs)
        elif len(args) > 0 or len(kwargs) > 0:
            cls._instance.__init__(*args,**kwargs)
        return cls._instance

    def __init__(self, training_matrix, undirected_category_tree, cat_div_method, poi_cats):
        self.training_matrix = training_matrix
        self.undirected_category_tree = undirected_category_tree
        self.cat_div_method = cat_div_method

        self.CAT_METHODS = {
            "std_norm": self.cat_div_std_norm,
            # "mad_norm": self.cat_div_mad_norm,
            "ild": self.cat_div_ild,
            "raw_std": self.cat_div_raw_std,
            "num_cat": self.cat_div_num_cat,
            "binomial": self.cat_div_binomial,
            "poi_ild": self.cat_div_poi_ild,
            "inv_num_cat": self.cat_div_num_cat,
            "qtd_cat": self.cat_div_qtd_cat,
        }

        self.poi_cats = poi_cats

        self.users_categories_visits = cat_utils.get_users_cat_visits(self.training_matrix,
                                                                      self.poi_cats)

        # import scipy.stats
        # # print('mean walk',self.mean_walk)
        # print(scipy.stats.describe(list(map(len,self.users_categories_visits))))
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1,1)
        # ax.hist(list(map(len,self.users_categories_visits)))
        # fig.savefig('resultadotemp2.png')


        self.cat_div_propensity = None
    
    @classmethod
    def cat_div_raw_std(cls, uid):
        self = cls.getInstance()
        cats_visits = self.users_categories_visits[uid]
        cats_visits = np.array(list(cats_visits.values()), dtype=np.int32)
        std = np.std(cats_visits)
        if std == 0:
            return 0
        else:
            return 2*std/(np.max(cats_visits)-np.min(cats_visits))

    @classmethod
    def cat_div_std_norm(cls, uid):
        self = cls.getInstance()
        cats_visits = self.users_categories_visits[uid]
        cats_visits = np.array(list(cats_visits.values()), dtype=np.int32)
        std = np.std(cats_visits)
        if std == 0:
            return 0
        else:
            return 1-2*std/(np.max(cats_visits)-np.min(cats_visits))

    # @classmethod
    # def cat_div_mad_norm(cls, uid):
    #     self = cls.getInstance()
    #     cats_visits = self.users_categories_visits[uid]
    #     std = scipy.stats.median_absolute_deviation(cats_visits)
    #     if std == 0:
    #         return 0
    #     else:
    #         return 1-std/(np.max(cats_visits)-np.min(cats_visits))

    @classmethod
    def cat_div_skew(cls, uid):
        self = cls.getInstance()
        cats_visits = self.users_categories_visits[uid]
        pass

    @classmethod
    def cat_div_ild(cls, uid):
        self = cls.getInstance()
        cats_visits = self.users_categories_visits[uid]
        dis_sum = 0
        for cat1 in cats_visits.keys():
            for cat2 in cats_visits.keys():
                if cat1 != cat2:
                    dis_sum += geocat.category_dis_sim(
                        cat1, cat2, self.undirected_category_tree)
        length = len(cats_visits)
        # print(length,dis_sum/(length**2-length))
        return dis_sum/(length**2-length)

    @classmethod
    def cat_div_num_cat(cls, uid):
        self = cls.getInstance()
        cats_visits = self.users_categories_visits[uid]
        # return len(cats_visits)/(len(self.undirected_category_tree)-1)
        return len(cats_visits)

    @classmethod
    def cat_div_inv_num_cat(cls, uid):
        self = cls.getInstance()
        cats_visits = self.users_categories_visits[uid]
        # return len(cats_visits)/(len(self.undirected_category_tree)-1)
        return len(cats_visits)

    @classmethod
    def cat_div_qtd_cat(cls, uid):
        self = cls.getInstance()
        cats_visits = self.users_categories_visits[uid]
        return len(cats_visits)/(len(self.undirected_category_tree)-1)

    # @classmethod
    # def cat_div_binomial(cls):
    #     self = cls.getInstance()
    #     pass

    @classmethod
    def cat_div_poi_ild(cls, uid):
        self = cls.getInstance()
        lids = self.training_matrix[uid].nonzero()[0]
        ild = metrics.ildk(lids,self.poi_cats,self.undirected_category_tree)
        return ild


    def compute_div_propensity(self,div_weight=0.75,alpha=0.0):
        # switcher = {
        #     "cat_div_std_norm": self.cat_div_std_norm,
        #     "cat_div_skew": self.cat_div_skew,
        #     "cat_div_mad_norm":self.cat_div_mad_norm,
        #     "cat_div_ld":self.cat_div_ld,
        # }
        if self.cat_div_method == 'binomial':
            self.binomial = Binomial.getInstance(self.training_matrix, self.poi_cats,
                                div_weight, alpha)
            self.binomial.compute_all_probabilities()
        if self.cat_div_method in ['num_cat','inv_num_cat']:
            self.poi_cats_median = np.median(list(map(len,self.poi_cats.values())))
            self.visits_median = np.median(np.sum(self.training_matrix,axis=1))
            self.num_cat_norm_value = self.poi_cats_median*self.visits_median
            
        func = self.CAT_METHODS.get(self.cat_div_method,
                                    lambda: "Invalid method")
        # args=[(i,) for i in self.users_categories_visits]

        # if self.cat_div_method == 'binomial' or self.cat_div_method == 'poi_ild':
        args=[(uid,) for uid in range(len(self.users_categories_visits))]
        self.cat_div_propensity = run_parallel(func, args, self.CHKS)
        self.cat_div_propensity = np.array(self.cat_div_propensity)
         
        if self.cat_div_method == 'num_cat':
            self.cat_div_propensity = np.clip(self.cat_div_propensity,None,self.num_cat_norm_value)/self.num_cat_norm_value
        elif self.cat_div_method == 'inv_num_cat':
            self.cat_div_propensity = 1-np.clip(self.cat_div_propensity,None,self.num_cat_norm_value)/self.num_cat_norm_value
        if self.cat_div_method == 'qtd_cat':
            self.cat_div_propensity = self.cat_div_propensity/np.max(self.cat_div_propensity)
        # bins = np.append(np.arange(0,1,1/(3-1)),1)
        # centers = (bins[1:]+bins[:-1])/2
        # self.cat_div_propensity = bins[np.digitize(self.cat_div_propensity, centers)]
        # executor = ProcessPoolExecutor()
        # futures = []
        # for cat_visits in self.users_categories_visits:
        #     futures.append(executor.submit(func, cat_visits))
        # self.cat_div_propensity = [futures[i].result()
        #                            for i in tqdm(range(len(futures)), desc='CatDivProp')]
        return self.cat_div_propensity

    @classmethod
    def cat_div_binomial(cls,uid):
        self = cls.getInstance()
        lids = self.training_matrix[uid].nonzero()[0]
        cats = set()
        for lid in lids:
            cats.update(self.poi_cats[lid])
        num_lids = len(lids)
        return 1-self.binomial.binom_div(uid, lids, num_lids, cats)
