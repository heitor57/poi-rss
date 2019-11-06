import os
import sys
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)
# print(sys.path)

from abc import ABC, abstractmethod
from collections import defaultdict
import pickle
from concurrent.futures import ProcessPoolExecutor
import json
import time

import numpy as np

import cat_utils
from usg.UserBasedCF import UserBasedCF
from usg.FriendBasedCF import FriendBasedCF
from usg.PowerLaw import PowerLaw
import geocat.objfunc as gcobjfunc


def normalize(scores):
    max_score = max(scores)
    if not max_score == 0:
        scores = [s / max_score for s in scores]
    return scores


class RecRunner:
    BASE_RECOMMENDERS = {
        "mostpopular": mostpopular,
        "usg": usg
    }
    FINAL_RECOMMENDERS = {
        "geocat": geocat,
    }

    def __init__(self, base_rec, final_rec, city,
                 base_rec_list_size, final_rec_list_size, data_directory):
        if base_rec not in self.BASE_RECOMMENDERS:
            self.base_rec = next(iter(self.BASE_RECOMMENDERS))
            print(f"Base recommender not detected, using default:{self.base_rec}")
        else:
            self.base_rec = base_rec
        if final_rec not in self.FINAL_RECOMMENDERS:
            self.final_rec = next(iter(self.FINAL_RECOMMENDERS))
            print(f"Base recommender not detected, using default:{self.final_rec}")
        else:
            self.final_rec = final_rec
        
        self.city = city

        self.base_rec_list_size = base_rec_list_size
        self.final_rec_list_size = final_rec_list_size
        if data_directory[-1] != '/':
            data_directory += '/'
        self.data_directory = data_directory

    def load_base(self):
        CITY = self.city
        print(f"{CITY} city base loading")
        # Train load
        self.data_checkin_train = pickle.load(
            open(self.data_directory+"checkin/train/"+CITY+".pickle", "rb"))

        # Test load
        self.ground_truth = defaultdict(set)
        for checkin in pickle.load(open(self.data_directory+"checkin/test/"+CITY+".pickle", "rb")):
            self.ground_truth[checkin['user_id']].add(checkin['poi_id'])
        # Pois load
        self.poi_coos = {}
        self.poi_cats = {}
        for poi_id, poi in pickle.load(open(self.data_directory+"poi/"+CITY+".pickle", "rb")).items():
            self.poi_coos[poi_id] = tuple([poi['latitude'], poi['longitude']])
            self.poi_cats[poi_id] = poi['categories']

        # Social relations load
        self.social_relations = defaultdict(list)
        for user_id, friends in pickle.load(open(self.data_directory+"user/"+CITY+".pickle", "rb")).items():
            self.social_relations[user_id] = friends

        user_num = len(self.social_relations)
        poi_num = len(self.poi_coos)
        user_num, poi_num

        # Cat Hierarchy load
        self.dict_alias_title, self.category_tree, self.dict_alias_depth = cat_utils.cat_structs(
            self.data_directory+"categories.json")
        self.undirected_category_tree = self.category_tree.to_undirected()

        # Training matrix create
        self.training_matrix = np.zeros((user_num, poi_num))
        for checkin in self.data_checkin_train:
            self.training_matrix[checkin['user_id'], checkin['poi_id']] += 1

        # poi neighbors load
        self.poi_neighbors = pickle.load(
            open(self.data_directory+"neighbor/"+CITY+".pickle", "rb"))
        print(f"{CITY} city base loaded")
        self.all_uids = list(range(user_num))
        self.all_lids = list(range(poi_num))

    def run_usg(self, U, S, G, uid, alpha, beta):
        if uid in self.ground_truth:

            U_scores = normalize([U.predict(uid, lid)
                                  if self.training_matrix[uid, lid] == 0 else -1
                                  for lid in self.all_lids])
            S_scores = normalize([S.predict(uid, lid)
                                  if self.training_matrix[uid, lid] == 0 else -1
                                  for lid in self.all_lids])
            G_scores = normalize([G.predict(uid, lid)
                                  if self.training_matrix[uid, lid] == 0 else -1
                                  for lid in self.all_lids])

            U_scores = np.array(U_scores)
            S_scores = np.array(S_scores)
            G_scores = np.array(G_scores)

            overall_scores = (1.0 - alpha - beta) * U_scores + \
                alpha * S_scores + beta * G_scores

            predicted = list(reversed(overall_scores.argsort()))[
                :self.base_rec_list_size]
            overall_scores = list(reversed(np.sort(overall_scores)))[
                :self.base_rec_list_size]
            #actual = ground_truth[uid]
            print(uid)
            return json.dumps({'user_id': uid, 'predicted': list(map(int, predicted)), 'scores': list(map(float, overall_scores))})+"\n"
        return None

    def usg(self):
        training_matrix = self.training_matrix
        all_uids = self.all_uids
        social_relations = self.social_relations
        poi_coos = self.poi_coos
        top_k = self.base_rec_list_size
        training_matrix = training_matrix.copy()
        training_matrix[training_matrix >= 1] = 1
        alpha = 0.1
        beta = 0.1

        U = UserBasedCF()
        S = FriendBasedCF(eta=0.05)
        G = PowerLaw()

        U.pre_compute_rec_scores(training_matrix)
        S.compute_friend_sim(social_relations, training_matrix)
        G.fit_distance_distribution(training_matrix, poi_coos)

        executor = ProcessPoolExecutor()
        print("Running usg")
        futures = [executor.submit(
            self.run_usg, U, S, G, uid, alpha, beta) for uid in all_uids]
        results = [future.result() for future in futures]
        print("usg terminated")
        # dview.map_sync(run_usg,range(user_num))

        result_out = open(self.data_directory+"result/reclist/" +
                          self.city+"_sigir11_top_" + str(top_k) + ".json", 'w')
        for json_string_result in results:
            result_out.write(json_string_result)
        result_out.close()
        print("Saved Rec List in "+self.data_directory+"result/reclist/" +
              self.city+"_sigir11_top_" + str(top_k) + ".json")

    def mostpopular(self):

        executor = ProcessPoolExecutor()

        futures = [executor.submit(self.run_mostpopular, uid)
                   for uid in self.all_uids]
        results = [future.result() for future in futures]

        result_out = open(self.data_directory+"result/reclist/"+self.city +
                          "_mostpopular_" + str(self.base_rec_list_size) + ".json", 'w')
        for json_string_result in results:
            result_out.write(json_string_result)
        result_out.close()

    def run_mostpopular(self, uid):
        if uid in self.ground_truth:
            poi_indexes = set(list(range(self.training_matrix.shape[1])))
            visited_indexes = set(self.training_matrix[uid].nonzero()[0])
            not_visited_indexes = poi_indexes-visited_indexes
            not_visited_indexes = np.array(list(not_visited_indexes))
            poi_visits_nu = np.count_nonzero(self.training_matrix, axis=0)
            pois_score = poi_visits_nu/self.training_matrix.shape[0]
            for i in visited_indexes:
                pois_score[i] = 0
            predicted = list(reversed(np.argsort(pois_score)))[
                0:self.base_rec_list_size]
            overall_scores = list(reversed(np.sort(pois_score)))[
                0:self.base_rec_list_size]
            return json.dumps({'user_id': uid, 'predicted': list(map(int, predicted)), 'scores': list(map(float, overall_scores))})+"\n"
        return None

    def run_geocat(self, uid):
        if uid in self.ground_truth:
            pois_score = self.run_mostpopular(uid)
            predicted = list(reversed(np.argsort(pois_score)))[
                0:self.base_rec_list_size]
            overall_scores = list(reversed(np.sort(pois_score)))[
                0:self.base_rec_list_size]

            actual = self.ground_truth[uid]
            start_time = time.time()

            predicted, overall_scores = gcobjfunc.geocat(uid, self.training_matrix, predicted, overall_scores, actual,
                                                         self.poi_cats, self.poi_neighbors, self.final_rec_list_size, self.undirected_category_tree)

            print("uid → %d, time → %fs" % (uid, time.time()-start_time))

            predicted = np.array(predicted)[list(
                reversed(np.argsort(overall_scores)))]
            overall_scores = list(reversed(np.sort(overall_scores)))

            return json.dumps({'user_id': uid, 'predicted': list(map(int, predicted)), 'scores': list(map(float, overall_scores))})+"\n"
        return None

    def geocat(self):

        executor = ProcessPoolExecutor()

        futures = [executor.submit(self.run_geocat, uid)
                   for uid in self.all_uids]
        results = [future.result() for future in futures]

        result_out = open(self.data_directory+"result/reclist/"+self.city +
                          "_geocat_" + str(self.base_rec_list_size) + ".json", 'w')
        for json_string_result in results:
            result_out.write(json_string_result)
        result_out.close()

    def run_base_recommender(self):

        pass
