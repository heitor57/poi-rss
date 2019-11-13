import numpy as np
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.713.513&rep=rep1&type=pdf
# 2014 vargas binomial
import scipy.stats
from collections import defaultdict

from tqdm import tqdm

class Binomial:
    def __init__(self,training_matrix,poi_cats,div_weight=0.75,alpha=0.5):
        self.training_matrix=training_matrix
        self.poi_cats=poi_cats
        self.div_weight=div_weight
        self.alpha = alpha # global/user tradeoff in coverage
        cats = set()
        for catss in list(poi_cats.values()):
            cats.update(catss)
        self.genres=cats
        # assert 'landscapearchitects' in self.genres
        # assert 'acupuncture' in self.genres
        # probability user genre
        self.p_u_g = defaultdict(dict) # user probability
        self.p_g = defaultdict() # global probability
        self.p = defaultdict(dict) # final probability
    # itens of s that has the g genre
    def k_g_s(self,genre,items):
        count=0
        for item in items:
            
            if genre in self.poi_cats[item]:
                count+=1
        return count

    # p^{''}_{g}
    def user_probability_genre(self,uid,genre):
        lids=self.training_matrix[uid].nonzero()[0]
        count=self.k_g_s(genre,lids)
        # for lid in lids:
        #     if genre in poi_cats[lid]:
        #         count+=1
        return count/len(lids)


    # p^{'}_{g}
    def global_probability_genre(self,genre):
        num_item_users_consumed=0
        count=0
        for uid in range(self.training_matrix.shape[0]):
            lids=self.training_matrix[uid,:].nonzero()[0]
            
            count+=self.k_g_s(genre,lids)
            num_item_users_consumed+=len(lids)
        return count/num_item_users_consumed

    def genre_probability(self,uid,genre):
        return (1-self.alpha)*self.p_u_g[uid][genre]+\
            self.alpha*self.p_g[genre]


    def compute_all_probabilities(self):
        print("Computing global probability")
        for genre in tqdm(self.genres):
            self.p_g[genre]=self.global_probability_genre(genre)
        print("Computing user probability")
        for uid in tqdm(range(self.training_matrix.shape[0])):
            lids=self.training_matrix[uid].nonzero()[0]
            genres=self.get_genres_in_rec_list(lids)
            for genre in self.genres:
                self.p_u_g[uid][genre]=0
                self.p[uid][genre]=0
            for genre in genres:
                self.p_u_g[uid][genre]=self.user_probability_genre(uid,genre)
                self.p[uid][genre]=self.genre_probability(uid,genre)

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
        
        return genres_in_rec_list

    def coverage(self,uid,rec_list,rec_list_size,rec_list_genres):
        # scipy.stats.binom(n=rec_list_size,
        #                     p=genre_probability(uid,genre,training_matrix,poi_cats,alpha))
        #genres=self.genres.copy()
        genres_in_rec_list=rec_list_genres
        exponent=1/len(self.genres)
        genres = self.genres-genres_in_rec_list


        cov=1
        for genre in genres:
            binom=scipy.stats.binom.pmf(n=rec_list_size,
                        p=self.p[uid][genre],k=0)
            cov*=binom**exponent
        return cov

    # def pmf_greater_equal(k):
    #     for l in 
    #     pass



    def non_redundancy(self,uid,rec_list,rec_list_size,rec_list_genres):
        genres=rec_list_genres
        num_genres = len(genres)
        result_non_red=1
        if num_genres>0:
            exponent=1/num_genres
            
            for genre in genres:
                k=self.k_g_s(genre,rec_list)

                binom=0
                for l in range(1,k-1):
                    binom+=scipy.stats.binom.pmf(n=rec_list_size,
                                                p=self.p[uid][genre],k=l)
                binom=(1-binom)**exponent
                result_non_red*=binom
        return result_non_red
        
    def binom_div(self,uid,rec_list,rec_list_size,rec_list_genres):
        return self.coverage(uid,rec_list,rec_list_size,rec_list_genres)*\
            self.non_redundancy(uid,rec_list,rec_list_size,rec_list_genres)

    def objective_function(self,uid,poi_id,score,rec_list,rec_list_size,rec_list_genres):
        crlg = rec_list_genres.copy()
        crlg.update(set(self.poi_cats[poi_id]))
        div=self.binom_div(uid,rec_list+[poi_id],rec_list_size,crlg)\
                            -self.binom_div(uid,rec_list,rec_list_size,rec_list_genres)
        return (1-self.div_weight)*score+self.div_weight*div

    def binomial(self,uid,tmp_rec_list,tmp_score_list,K):
        
        range_K=range(K)
        rec_list=[]
        
        final_scores=[]
        rec_list_genres=set()
        for i in range_K:
            #print(i)
            poi_to_insert=None
            max_objective_value=-200
            print("ejwiq")
            for j in range(len(tmp_rec_list)):
                candidate_poi_id=tmp_rec_list[j]
                candidate_score=tmp_score_list[j]
                objective_value=self.objective_function(uid,candidate_poi_id,candidate_score,
                                                        rec_list,K,rec_list_genres)
                if objective_value > max_objective_value:
                    max_objective_value=objective_value
                    poi_to_insert=candidate_poi_id
            if poi_to_insert is not None:
                rm_idx=tmp_rec_list.index(poi_to_insert)
                tmp_rec_list.pop(rm_idx)
                tmp_score_list.pop(rm_idx)
                rec_list.append(poi_to_insert)
                final_scores.append(max_objective_value)
                # update genres of reclist
                rec_list_genres.update(set(self.poi_cats[poi_to_insert]))
        return rec_list,final_scores