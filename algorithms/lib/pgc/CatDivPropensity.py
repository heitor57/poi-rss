from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy
from progressbar import progressbar

import geocat.objfunc as geocat


class CatDivPropensity():
    def __init__(self,training_matrix,users_categories_visits,undirected_category_tree,cat_div_method='std_norm'):
        self.training_matrix=training_matrix
        self.users_categories_visits=users_categories_visits
        self.undirected_category_tree=undirected_category_tree
        self.cat_div_method=cat_div_method

        self.CAT_METHODS = {
            "std_norm": self.cat_div_std_norm,
            "mad_norm": self.cat_div_mad_norm,
            "ld": self.cat_div_ld
        }

        self.cat_div_propensity=None
    def cat_div_std_norm(self,cats_visits):
        cats_visits=np.array(list(cats_visits.values()),dtype=np.int32)
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
        for cat_visits in self.users_categories_visits:
            futures.append(executor.submit(func,cat_visits))
        self.cat_div_propensity = [futures[i].result() for i in progressbar(range(len(futures)))]
        return self.cat_div_propensity

