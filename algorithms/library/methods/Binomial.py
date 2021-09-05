# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.713.513&rep=rep1&type=pdf
# 2014 vargas binomial

import numpy as np
import scipy.special
import scipy.stats

from collections import defaultdict
from time import time
from tqdm import tqdm
import multiprocessing

from parallel_util import run_parallel


class Binomial:
    _instance = None

    @classmethod
    def getInstance(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = cls(*args, **kwargs)
        elif len(args) > 0 or len(kwargs) > 0:
            cls._instance.__init__(*args, **kwargs)
        return cls._instance

    def __init__(self, training_matrix, poi_cats, cat_num, div_weight=0.75, alpha=0.5):
        self.training_matrix = training_matrix
        self.poi_cats = poi_cats
        self.div_weight = div_weight
        self.alpha = alpha  # global/user tradeoff in coverage
        cats = set()
        for catss in list(poi_cats.values()):
            cats.update(catss)
        self.genres = cats
        # assert 'landscapearchitects' in self.genres
        # assert 'acupuncture' in self.genres
        # probability user genre
        self.p_u_g = defaultdict(dict)  # user probability
        self.p_g = defaultdict()  # global probability
        self.p = defaultdict(dict)  # final probability

        self.poi_cat_matrix = np.zeros((len(poi_cats), cat_num), dtype=bool)
        for lid, cats in poi_cats.items():
            self.poi_cat_matrix[lid, cats] = True
    # itens of s that has the g genre

    def k_g_s(self, genre, items):
        # count=0
        # for item in items:
        #     if genre in self.poi_cats[item]:
        #         count+=1
        # return count
        return np.count_nonzero(self.poi_cat_matrix[items, :][:, genre], axis=0)

    # p^{''}_{g}
    def user_probability_genre(self, uid, genre):
        lids = self.training_matrix[uid].nonzero()[0]
        count = self.k_g_s(genre, lids)
        # for lid in lids:
        #     if genre in poi_cats[lid]:
        #         count+=1
        return count/len(lids)

    # deprecated !!!
    # p^{'}_{g}

    @classmethod
    def global_probability_genre(cls, genre):
        self = cls.getInstance()
        # num_item_users_consumed = np.count_nonzero(self.training_matrix)
        # count=np.count_nonzero(self.training_matrix*self.poi_cat_matrix[:,genre])
        num_item_users_consumed = 0
        count = 0
        for uid in range(self.training_matrix.shape[0]):
            lids = self.training_matrix[uid, :].nonzero()[0]

            count += self.k_g_s(genre, lids)
            num_item_users_consumed += len(lids)
        return count/num_item_users_consumed

    def genre_probability(self, uid, genre):
        return (1-self.alpha)*self.p_g[genre] +\
            self.alpha*self.p_u_g[uid][genre]

    @classmethod
    def compute_user_genre_probability(cls, uid):
        self = cls.getInstance()
        lids = self.training_matrix[uid].nonzero()[0]
        genres = self.get_genres_in_rec_list(lids)
        user_probability_genre = dict()
        # genre_probability = dict()
        for genre in self.genres:
            user_probability_genre[genre] = 0
            # genre_probability[genre]=0
        for genre in genres:
            user_probability_genre[genre] = self.user_probability_genre(
                uid, genre)
            # genre_probability[genre]=self.genre_probability(uid,genre)
        return user_probability_genre  # , genre_probability

    @classmethod
    def compute_genre_probability(cls, uid):
        self = cls.getInstance()
        lids = self.training_matrix[uid].nonzero()[0]
        genres = self.get_genres_in_rec_list(lids)
        # user_probability_genre = dict()
        genre_probability = dict()
        for genre in self.genres:
            # user_probability_genre[genre]=0
            genre_probability[genre] = 0
        for genre in genres:
            # user_probability_genre[genre]=self.user_probability_genre(uid,genre)
            genre_probability[genre] = self.genre_probability(uid, genre)
        return genre_probability

    def compute_all_probabilities(self):
        print("Computing global probability")
        num_cores = multiprocessing.cpu_count()
        num_divs = 4
        chk_size_genres = int(len(self.genres)/num_cores/num_divs)
        chk_size_users = int(self.training_matrix.shape[0]/num_cores/num_divs)
        # args = [(genre,) for genre in self.genres]
        # result = run_parallel(self.global_probability_genre,args,chk_size_genres)
        stime = time()
        num_item_users_consumed = np.count_nonzero(self.training_matrix)
        result = np.sum(np.matmul(np.where(self.training_matrix > 1, 1,
                        self.training_matrix), self.poi_cat_matrix), axis=0)/num_item_users_consumed
        for genre, p_g in enumerate(result):
            self.p_g[genre] = p_g
        print(f'{time()-stime}s')
        # for genre, p_g in zip(self.genres,result):
        #     self.p_g[genre]= p_g
        # for genre in tqdm(self.genres):
        #     self.p_g[genre]=self.global_probability_genre(genre)
        print("\tComputing user probability")
        args = [(uid,) for uid in range(self.training_matrix.shape[0])]
        print("Computing user probability genre")
        result = run_parallel(
            self.compute_user_genre_probability, args, chk_size_users)
        for uid, p_u_g in zip(range(self.training_matrix.shape[0]), result):
            self.p_u_g[uid] = p_u_g

        print("Computing genre probability")
        result = run_parallel(
            self.compute_genre_probability, args, chk_size_users)
        for uid, p in zip(range(self.training_matrix.shape[0]), result):
            self.p[uid] = p

    # def binomial_probability(N,k,p):
    #     """
    #     Keyword arguments:
    #     N -- rec list size normally, or number of trials
    #     k -- number of success
    #     p -- probability of success
    #     """

    def get_genres_in_rec_list(self, rec_list):
        genres_in_rec_list = set()
        for lid in rec_list:
            genres_in_rec_list.update(self.poi_cats[lid])
        return genres_in_rec_list

    # def probability_mass_function(self,n,k,p):
    #     return scipy.special.comb(n,k)*p**k*(1-p)**(n-k)

    def coverage(self, uid, rec_list, rec_list_size, rec_list_genres):

        # scipy.stats.binom(n=rec_list_size,
        #                     p=genre_probability(uid,genre,training_matrix,poi_cats,alpha))
        # genres=self.genres.copy()
        genres_in_rec_list = rec_list_genres
        exponent = 1/len(self.genres)
        genres = self.genres-genres_in_rec_list

        cov = 1

        cov = np.prod(scipy.stats.binom.pmf(n=rec_list_size,
                                            p=[self.p[uid][genre] for genre in genres], k=0)**exponent)
        # for genre in genres:
        #     binom=scipy.stats.binom.pmf(n=rec_list_size,
        #                p=self.p[uid][genre],k=0)
        #     # binom = self.probability_mass_function(rec_list_size,0,self.p[uid][genre])
        #     cov*=binom**exponent
        return cov

    def non_redundancy(self, uid, rec_list, rec_list_size, rec_list_genres):

        genres = rec_list_genres
        num_genres = len(genres)
        result_non_red = 1
        if num_genres > 0:
            exponent = 1/num_genres
            ks = self.k_g_s(list(genres), rec_list)

            for i, genre in enumerate(genres):
                # k=self.k_g_s(genre,rec_list)
                k = ks[i]

                # binom=0
                # for l in range(1,k-1):
                #     binom+=scipy.stats.binom.pmf(n=rec_list_size,
                #                                 p=self.p[uid][genre],k=l)
                binom = np.sum(scipy.stats.binom.pmf(n=rec_list_size,
                                                     p=self.p[uid][genre], k=list(range(1, k)))
                               )
                # binom+=self.probability_mass_function(rec_list_size,l,self.p[uid][genre])
                binom = (1-binom)**exponent
                result_non_red *= binom

        return result_non_red

    def binom_div(self, uid, rec_list, rec_list_size, rec_list_genres):
        # stime = time()
        # cov = self.coverage(uid,rec_list,rec_list_size,rec_list_genres)
        # print(f'cov {time()-stime}s')
        # stime = time()
        # non_red = self.non_redundancy(uid,rec_list,rec_list_size,rec_list_genres)
        # print(f'non_red {time()-stime}s')
        # return cov*non_red
        return self.coverage(uid, rec_list, rec_list_size, rec_list_genres) *\
            self.non_redundancy(uid, rec_list, rec_list_size, rec_list_genres)

    def objective_function(self, uid, poi_id, score, rec_list, rec_list_size, rec_list_genres, current_binom_div):
        crlg = rec_list_genres.copy()
        crlg.update(set(self.poi_cats[poi_id]))
        div = self.binom_div(uid, rec_list+[poi_id], rec_list_size, crlg)\
            - current_binom_div
        return (1-self.div_weight)*score+self.div_weight*div

    def binomial(self, uid, tmp_rec_list, tmp_score_list, K):

        range_K = range(K)
        rec_list = []

        final_scores = []
        rec_list_genres = set()
        current_binom_div = self.binom_div(uid, rec_list, K, rec_list_genres)

        for i in range_K:
            # print(i)
            poi_to_insert = None
            max_objective_value = -200

            for j in range(len(tmp_rec_list)):
                candidate_poi_id = tmp_rec_list[j]
                candidate_score = tmp_score_list[j]
                objective_value = self.objective_function(uid, candidate_poi_id, candidate_score,
                                                          rec_list, K, rec_list_genres, current_binom_div)
                if objective_value > max_objective_value:
                    max_objective_value = objective_value
                    poi_to_insert = candidate_poi_id
            if poi_to_insert is not None:
                rm_idx = tmp_rec_list.index(poi_to_insert)
                tmp_rec_list.pop(rm_idx)
                tmp_score_list.pop(rm_idx)
                rec_list.append(poi_to_insert)
                final_scores.append(max_objective_value)
                # update genres of reclist
                rec_list_genres.update(set(self.poi_cats[poi_to_insert]))
                # update current binom div
                current_binom_div = self.binom_div(
                    uid, rec_list, K, rec_list_genres)
        return rec_list, final_scores
