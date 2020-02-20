import numpy as np
import scipy.special
import scipy.stats
from collections import defaultdict
from time import time
from tqdm import tqdm

class Pm2:
    def __init__(self,training_matrix,poi_cats,main_quotient_weight):
        self.training_matrix=training_matrix
        self.num_users = self.training_matrix.shape[0]
        self.num_items = self.training_matrix.shape[1]
        self.poi_cats=poi_cats
        cats = set()
        for catss in list(poi_cats.values()):
            cats.update(catss)
        self.cats = list(cats)
        self.num_cats = len(cats)
        self.main_quotient_weight = main_quotient_weight


        self.cat_to_id = dict()
        for i, cat in enumerate(self.cats):
            self.cat_to_id[cat] = i
        
        self.visits_cats_matrix = np.zeros((self.num_users,self.num_cats))
        print("Creating categories visits matrix...")
        for uid in range(self.num_users):
            lids = np.nonzero(self.training_matrix[uid])[0]
            for lid in lids:
                cats = poi_cats[lid]
                for cat in cats:
                    cat_index = self.cat_to_id[cat]
                    self.visits_cats_matrix[uid,cat_index] += 1
        print("Categories visits matrix created.")


        # self.items_cats_matrix = np.zeros((self.num_items,self.num_cats))
        # print("Creating pois x categories (relevance in cat) matrix...")
        # for lid in range(self.num_items):
        #     for cat in poi_cats[lid]:
        #         cat_index = self.cat_to_id[cat]
        #         self.items_cats_matrix[lid,cat_index] += 1
        # for cid in range(self.num_cats):
        #     self.items_cats_matrix[:,cid] = self.items_cats_matrix[:,cid]/np.count_nonzero(self.items_cats_matrix[:,cid])
        # print("pois x categories (relevance in cat) matrix created.")
        
        

    def getUserCategoryProbability(self,uid):
        sum_value = np.sum(self.visits_cats_matrix[uid])
        if sum_value == 0:
            print(f"User {uid} without categories visits")
        if sum_value == 0:
            prob = np.zeros(self.num_cats)
            prob = prob+1/self.num_cats
        else:
            prob = self.visits_cats_matrix[uid]/sum_value

        return prob

    @staticmethod
    @np.vectorize
    def sainteLagueQuotient(v, s):
        return v/(2*s + 1)

    def objective_function(self, candidate_id, score, quotient, i_star):
        sub_sum = 0
        main_sum = 0
        for cat in self.poi_cats[candidate_id]:
            cat_id = self.cat_to_id[cat]
            if cat_id != i_star:
                sub_sum += score * quotient[cat_id]
            else:
                main_sum += score * quotient[i_star]
        #sub_sum = np.sum(quotient*score,where=list(range(self.num_cats))!=i_star)
        return self.main_quotient_weight*main_sum+(1-self.main_quotient_weight)*sub_sum
    
    def pm2(self,uid,tmp_rec_list,tmp_score_list,K):
        # from pudb import set_trace; set_trace()
        #sainteLagueQuotient = np.vectorize(self.sainteLagueQuotient)
        quotient = np.zeros(self.num_cats)
        # multiply probability with rec list size
        v = self.getUserCategoryProbability(uid)*K

        s = np.zeros(self.num_cats)
        rec_list=[]
        final_scores=[]

        for i in range(K):
            max_quotient = 0
            quotient = self.sainteLagueQuotient(v,s)
            # category with max value
            i_star = np.argmax(quotient)
            num_cur_candidates = len(tmp_rec_list)

            poi_to_insert = None
            max_objective_value = -200

            for j in range(num_cur_candidates):
                candidate_poi_id = tmp_rec_list[j]
                candidate_score = tmp_score_list[j]
                objective_value = self.objective_function(candidate_poi_id,candidate_score,
                                                          quotient, i_star)
                if objective_value > max_objective_value:
                    max_objective_value=objective_value
                    poi_to_insert=candidate_poi_id
                    old_score = candidate_score

            if poi_to_insert is not None:
                rm_idx=tmp_rec_list.index(poi_to_insert)
                tmp_rec_list.pop(rm_idx)
                tmp_score_list.pop(rm_idx)
                rec_list.append(poi_to_insert)
                final_scores.append(max_objective_value)
                poi_num_cats = len(self.poi_cats[poi_to_insert])
                if poi_num_cats != 0:
                    if old_score != 0:
                        for cat in self.poi_cats[poi_to_insert]:
                            cat_id = self.cat_to_id[cat]
                            s[cat_id] += old_score/(old_score*poi_num_cats)
                #     else:
                #         print('PM2 selected poi with old score = 0 ?!?!?')
                # else:
                #     print('PM2 selected poi with no cats ?!?!?')
        return rec_list,final_scores
