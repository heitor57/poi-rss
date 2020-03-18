import time
import numpy as np
from collections import defaultdict
from parallel_util import run_parallel

class FriendBasedCF():
    _instance = None

    @classmethod
    def getInstance(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance=cls(*args,**kwargs)
        return cls._instance

    def __init__(self, eta=0.5):
        self.eta = eta
        self.social_proximity = defaultdict(list)
        self.check_in_matrix = None

    def compute_friend_sim(self, social_relations, check_in_matrix):
        ctime = time.time()
        print("Precomputing similarity between friends...", )
        self.check_in_matrix = check_in_matrix
        self.social_relations = social_relations

        args=[(uid,) for uid in self.social_relations]

        results = run_parallel(self.user_friend_sim,args,50)
        # for uid in self.social_relations:
        #     self.user_friend_sim(uid)
        for uid, social_proximity in results:
            self.social_proximity[uid] = social_proximity
        print("Done. Elapsed time:", time.time() - ctime, "s")

    @classmethod
    def user_friend_sim(cls,uid):
        self = cls.getInstance()
        social_proximity = []
        for fid in self.social_relations[uid]:
            if uid < fid:
                u_social_neighbors = set(self.social_relations[uid])
                f_social_neighbors = set(self.social_relations[fid])
                jaccard_friend = (1.0 * len(u_social_neighbors.intersection(f_social_neighbors)) /
                                    len(u_social_neighbors.union(f_social_neighbors)))

                u_check_in_neighbors = set(self.check_in_matrix[uid, :].nonzero()[0])
                f_check_in_neighbors = set(self.check_in_matrix[fid, :].nonzero()[0])
                jaccard_check_in = (1.0 * len(u_check_in_neighbors.intersection(f_check_in_neighbors)) /
                                    len(u_check_in_neighbors.union(f_check_in_neighbors)))
                if jaccard_friend > 0 and jaccard_check_in > 0:
                    social_proximity.append([fid, jaccard_friend, jaccard_check_in])
        return uid, social_proximity

    def predict(self, i, j):
        if i in self.social_proximity:
            numerator = np.sum([(self.eta * jf + (1 - self.eta) * jc) * self.check_in_matrix[k, j]
                                for k, jf, jc in self.social_proximity[i]])
            return numerator
        return 0.0
