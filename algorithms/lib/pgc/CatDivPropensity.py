from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy
from tqdm import tqdm
from parallel_util import run_parallel

import geocat.objfunc as geocat
from geocat.Binomial import Binomial
import metrics


class CatDivPropensity():
    CHKS = 50 # chunk size for parallel pool executor
    _instance = None
    METHODS = ['std_norm','mad_norm','ld','raw_std','num','binomial','poi_ild']

    @classmethod
    def getInstance(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance=cls(*args,**kwargs)
        return cls._instance

    def __init__(self, training_matrix, users_categories_visits,
                 undirected_category_tree, cat_div_method, poi_cats):
        self.training_matrix = training_matrix
        self.users_categories_visits = users_categories_visits
        self.undirected_category_tree = undirected_category_tree
        self.cat_div_method = cat_div_method

        self.CAT_METHODS = {
            "std_norm": self.cat_div_std_norm,
            "mad_norm": self.cat_div_mad_norm,
            "ld": self.cat_div_ld,
            "raw_std": self.cat_div_raw_std,
            "num": self.cat_div_num,
            "binomial": self.cat_div_binomial,
            "poi_ild": self.cat_div_poi_ild,
        }

        self.poi_cats = poi_cats

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

    @classmethod
    def cat_div_mad_norm(cls, uid):
        self = cls.getInstance()
        cats_visits = self.users_categories_visits[uid]
        std = scipy.stats.median_absolute_deviation(cats_visits)
        if std == 0:
            return 0
        else:
            return 1-std/(np.max(cats_visits)-np.min(cats_visits))

    @classmethod
    def cat_div_skew(cls, cats_visits):
        self = cls.getInstance()
        cats_visits = self.users_categories_visits[uid]
        pass

    @classmethod
    def cat_div_ld(cls, cats_visits):
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
    def cat_div_num(cls,cats_visits):
        self = cls.getInstance()
        cats_visits = self.users_categories_visits[uid]
        return len(cats_visits)/(len(self.undirected_category_tree)-1)
    # @classmethod
    # def cat_div_binomial(cls):
    #     self = cls.getInstance()
    #     pass

    @classmethod
    def cat_div_poi_ild(cls,uid):
        self = cls.getInstance()
        lids = self.training_matrix[uid].nonzero()[0]
        ild = metrics.ildk(lids,self.poi_cats,self.undirected_category_tree)
        return ild

    def compute_cat_div_propensity(self,div_weight=0.75,alpha=0.5):
        # switcher = {
        #     "cat_div_std_norm": self.cat_div_std_norm,
        #     "cat_div_skew": self.cat_div_skew,
        #     "cat_div_mad_norm":self.cat_div_mad_norm,
        #     "cat_div_ld":self.cat_div_ld,
        # }
        if self.cat_div_method == 'binomial':
            self.binomial = Binomial(self.training_matrix, self.poi_cats,
                                div_weight, alpha)
            self.binomial.compute_all_probabilities()

        func = self.CAT_METHODS.get(self.cat_div_method,
                                    lambda: "Invalid method")
        # args=[(i,) for i in self.users_categories_visits]

        # if self.cat_div_method == 'binomial' or self.cat_div_method == 'poi_ild':
        args=[(uid,) for uid in range(len(self.users_categories_visits))]
        self.cat_div_propensity = run_parallel(func, args, self.CHKS)

        # executor = ProcessPoolExecutor()
        # futures = []
        # for cat_visits in self.users_categories_visits:
        #     futures.append(executor.submit(func, cat_visits))
        # self.cat_div_propensity = [futures[i].result()
        #                            for i in tqdm(range(len(futures)), desc='CatDivProp')]
        return np.array(self.cat_div_propensity)

    @classmethod
    def cat_div_binomial(cls,uid):
        self = cls.getInstance()
        lids = self.training_matrix[uid].nonzero()[0]
        cats = set()
        for lid in lids:
            cats.update(self.poi_cats[lid])
        num_lids = len(lids)
        return self.binomial.binom_div(uid, lids, num_lids, cats)
