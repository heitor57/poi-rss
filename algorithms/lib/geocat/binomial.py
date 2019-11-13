import numpy as np
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.713.513&rep=rep1&type=pdf
# 2014 vargas binomial
import scipy.stats
from collections import defaultdict
from progressbar import progressbar
class Binomial:
    def __init__(self,training_matrix,poi_cats,div_weight=0.75,alpha=0.5):
        self.training_matrix=training_matrix
        self.poi_cats=poi_cats
        self.div_weight=div_weight
        self.alpha = alpha # global/user tradeoff in coverage
        cats = set()
        for cat in list(poi_cats.values()):
            cats.add(cat)
        self.genres=list(cats)
        # probability user genre
        self.p_u_g = defaultdict(dict)
        
    # itens of s that has the g genre
    def k_g_s(self,genre,items):
        count=0
        for item in items:
            if genre in self.poi_cats[item]:
                count+=1
        return count

    # p^{''}_{g}
    def user_probability_genre(self,uid,genre):
        lids=self.training_matrix[uid].nonzero()
        count=self.k_g_s(genre,lids)
        # for lid in lids:
        #     if genre in poi_cats[lid]:
        #         count+=1
        return count/len(lids)


    # p^{'}_{g}
    def global_probability_genre(self,genre):
        num_item_users_consumed=0
        count=0
        for uid in self.training_matrix.shape[0]:
            lids=self.training_matrix[uid].nonzero()
            count+=self.k_g_s(genre,lids)
            num_item_users_consumed+=len(lids)
        return count/num_item_users_consumed

    def genre_probability(self,uid,genre):
        return (1-self.alpha)*self.user_probability_genre(uid,genre)+\
            self.alpha*self.global_probability_genre(genre)


    def compute_all_probabilities(self):
        for uid in progressbar(range(self.training_matrix.shape[0])):
            lids=self.training_matrix[uid].nonzero()
            genres=self.get_genres_in_rec_list(lids)
            for genre in genres:
                self.p_u_g[uid][genre]=self.genre_probability(uid,genre)


    # def binomial_probability(N,k,p):
    #     """
    #     Keyword arguments:
    #     N -- rec list size normally, or number of trials
    #     k -- number of success
    #     p -- probability of success
    #     """

    def get_genres_in_rec_list(self,rec_list):
        genres_in_rec_list = set()
        for lid in rec_list:
            genres_in_rec_list.update(self.poi_cats[lid])
        genres_in_rec_list=list(genres_in_rec_list)
        return genres_in_rec_list

    def coverage(self,uid,rec_list,rec_list_size):
        # scipy.stats.binom(n=rec_list_size,
        #                     p=genre_probability(uid,genre,training_matrix,poi_cats,alpha))
        genres=self.genres.copy()
        genres_in_rec_list=self.get_genres_in_rec_list(rec_list)
        exponent=1/len(self.genres)
        for genre in genres_in_rec_list:
            genres.remove(genre)

        cov=1
        for genre in genres:
            binom=scipy.stats.binom.pmf(n=rec_list_size,
                        p=self.p_u_g[uid][genre],k=0)
            cov*=binom**exponent

    # def pmf_greater_equal(k):
    #     for l in 
    #     pass



    def non_redundancy(self,uid,rec_list,rec_list_size):
        genres=self.get_genres_in_rec_list(rec_list)
        exponent=1/len(genres)
        result_non_red=1
        for genre in genres:
            k=self.k_g_s(genre,rec_list)
            binom=0
            for l in range(1,k-1):
                binom+=scipy.stats.binom.pmf(n=rec_list_size,
                                            p=self.p_u_g[uid][genre],k=0)
            binom=(1-binom)**exponent
            result_non_red*=binom
        
    def binom_div(self,uid,rec_list,rec_list_size):
        return self.coverage(uid,rec_list,rec_list_size)*self.non_redundancy(uid,rec_list,rec_list_size)

    def objective_function(self,uid,score,rec_list,rec_list_size):
        div=self.binom_div(uid,rec_list,rec_list_size)
        return (1-self.div_weight)*score+self.div_weight*div

    def binomial(self,uid,tmp_rec_list,tmp_score_list,K):
        
        range_K=range(K)
        rec_list=[]
        
        final_scores=[]
        for i in range_K:
            #print(i)
            poi_to_insert=None
            max_objective_value=-200
            for j in range(len(tmp_rec_list)):
                candidate_poi_id=tmp_rec_list[j]
                candidate_score=tmp_score_list[j]
                objective_value=self.objective_function(uid,candidate_score,rec_list,K)
                if objective_value > max_objective_value:
                    max_objective_value=objective_value
                    poi_to_insert=candidate_poi_id
            if poi_to_insert is not None:
                rm_idx=tmp_rec_list.index(poi_to_insert)
                tmp_rec_list.pop(rm_idx)
                tmp_score_list.pop(rm_idx)
                rec_list.append(poi_to_insert)
                final_scores.append(max_objective_value)
        return rec_list,final_scores