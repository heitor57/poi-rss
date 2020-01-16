from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy
from tqdm import tqdm
from parallel_util import run_parallel

import geocat.objfunc as geocat


class CatDivPropensity():
    CHKS = 50 # chunk size for parallel pool executor
    _instance = None
    @classmethod
    def getInstance(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance=cls(*args,**kwargs)
        return cls._instance

    def __init__(self, training_matrix, users_categories_visits,
                 undirected_category_tree, cat_div_method='std_norm'):
        self.training_matrix = training_matrix
        self.users_categories_visits = users_categories_visits
        self.undirected_category_tree = undirected_category_tree
        self.cat_div_method = cat_div_method

        self.CAT_METHODS = {
            "std_norm": self.cat_div_std_norm,
            "mad_norm": self.cat_div_mad_norm,
            "ld": self.cat_div_ld,
            "raw_std": self.cat_div_raw_std,
        }

        self.cat_div_propensity = None

    @classmethod
    def cat_div_raw_std(cls, cats_visits):
        self = cls.getInstance()
        cats_visits = np.array(list(cats_visits.values()), dtype=np.int32)
        std = np.std(cats_visits)
        if std == 0:
            return 0
        else:
            return 2*std/(np.max(cats_visits)-np.min(cats_visits))

    @classmethod
    def cat_div_std_norm(cls, cats_visits):
        cats_visits = np.array(list(cats_visits.values()), dtype=np.int32)
        std = np.std(cats_visits)
        if std == 0:
            return 0
        else:
            return 1-2*std/(np.max(cats_visits)-np.min(cats_visits))

    @classmethod
    def cat_div_mad_norm(cls, cats_visits):
        std = scipy.stats.median_absolute_deviation(cats_visits)
        if std == 0:
            return 0
        else:
            return 1-std/(np.max(cats_visits)-np.min(cats_visits))

    @classmethod
    def cat_div_skew(cls, cats_visits):
        pass

    @classmethod
    def cat_div_ld(cls, cats_visits):
        self = cls.getInstance()
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
    def cat_div_binomial(cls):
        self = cls.getInstance()
        pass

    def compute_cat_div_propensity(self):
        # switcher = {
        #     "cat_div_std_norm": self.cat_div_std_norm,
        #     "cat_div_skew": self.cat_div_skew,
        #     "cat_div_mad_norm":self.cat_div_mad_norm,
        #     "cat_div_ld":self.cat_div_ld,
        # }
        func = self.CAT_METHODS.get(self.cat_div_method,
                                    lambda: "Invalid method")
        args=[(i,) for i in self.users_categories_visits]
        self.cat_div_propensity = run_parallel(func, args, self.CHKS)

        # executor = ProcessPoolExecutor()
        # futures = []
        # for cat_visits in self.users_categories_visits:
        #     futures.append(executor.submit(func, cat_visits))
        # self.cat_div_propensity = [futures[i].result()
        #                            for i in tqdm(range(len(futures)), desc='CatDivProp')]
        return np.array(self.cat_div_propensity)


    def cat_div_binomial(self, poi_cats, div_weight, alpha):
        binomial = Binomial(self.training_matrix, poi_cats,
            div_weight, alpha)
        binomial.compute_all_probabilities()
        for uid in range(len(self.training_matrix.shape[0])):

            lids = self.training_matrix[uid].nonzero()
            cats = {poi_cats[lid] for lid in lids}
            num_lids = len(lids)
            binomial.binom_div(uid, lids, num_lids, cats)
