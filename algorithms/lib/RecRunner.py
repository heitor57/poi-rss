# import os
# import sys
# module_path = os.path.abspath(os.path.join('.'))
# if module_path not in sys.path:
#     sys.path.append(module_path)
#print(sys.path)

from abc import ABC, abstractmethod
from collections import defaultdict, OrderedDict
import pickle
from concurrent.futures import ProcessPoolExecutor
import json
import time
from datetime import datetime
import itertools
import multiprocessing

import numpy as np
from tqdm import tqdm
from scipy.stats import describe
import matplotlib.pyplot as plt
from pympler import asizeof

import cat_utils
from usg.UserBasedCF import UserBasedCF
from usg.FriendBasedCF import FriendBasedCF
from usg.PowerLaw import PowerLaw
import geocat.objfunc as gcobjfunc
from pgc.GeoDivPropensity import GeoDivPropensity
from pgc.CatDivPropensity import CatDivPropensity
from constants import experiment_constants
import metrics
from geocat.Binomial import Binomial
from parallel_util import run_parallel

DATA_DIRECTORY = '../data'  # directory with all data

R_FORMAT = '.json'  # Results format is json, metrics, rec lists, etc
D_FORMAT = '.pickle'  # data set format is pickle

TRAIN = 'checkin/train'  # train data sets
TEST = 'checkin/test'  # test data sets
POI = 'poi/'  # poi data sets with cats and coos
USER = 'user/'  # users and friends
NEIGHBOR = 'neighbor/'  # neighbors of pois

METRICS = 'result/metrics/'
METRICS = 'result/reclist/'

#CHKS = 40 # chunk size for process pool executor
#CHKSL = 200 # chunk size for process pool executor largest

def normalize(scores):
    scores = np.array(scores, dtype=np.float128)

    max_score = np.max(scores)
    if not max_score == 0:
        scores = [s / max_score for s in scores]
    return scores


def dict_to_list_gen(d):
    for k, v in zip(d.keys(), d.values()):
        yield k
        yield v


def dict_to_list(d):
    return list(dict_to_list_gen(d))

def print_dict(dictionary):
    for key, value in dictionary.items():
        print(f"{key} : {value}")

class RecRunner():
    _instance = None
    def save_result(self,results,base=True):
        if base:
            result_out = open(self.data_directory+"result/reclist/" + self.get_base_rec_file_name(), 'w')
        else:
            result_out = open(self.data_directory+"result/reclist/" + self.get_final_rec_file_name(), 'w')
        for json_string_result in results:
            result_out.write(json_string_result)
        result_out.close()

    @classmethod
    def getInstance(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance=cls(*args,**kwargs)
        return cls._instance
        
    def __init__(self, base_rec, final_rec, city,
                 base_rec_list_size, final_rec_list_size, data_directory,
                 base_rec_parameters={}, final_rec_parameters={},except_final_rec=[]):
        self.BASE_RECOMMENDERS = {
            "mostpopular": self.mostpopular,
            "usg": self.usg
        }
        self.FINAL_RECOMMENDERS = {
            "geocat": self.geocat,
            "persongeocat": self.persongeocat,
            "geodiv": self.geodiv,
            "ld": self.ld,
            "binomial": self.binomial
        }
        # self.BASE_RECOMMENDERS_PARAMETERS = {
        #     "mostpopular": [],
        #     "usg": ['alpha','beta','eta']
        # }
        # self.FINAL_RECOMMENDERS_PARAMETERS = {
        #     "geocat": ['div_weight','div_geo_cat_weight'],
        #     "persongeocat": ['div_weight']
        # }
        self.base_rec = base_rec
        self.final_rec = final_rec

        self.base_rec_parameters = base_rec_parameters
        self.final_rec_parameters = final_rec_parameters

        self.city = city

        self.base_rec_list_size = base_rec_list_size
        self.final_rec_list_size = final_rec_list_size

        self.data_directory = data_directory

        # buffers de resultado do metodo base
        self.user_base_predicted_lid = {}
        self.user_base_predicted_score = {}
        # buffers de resultado do metodo final
        self.user_final_predicted_lid = {}
        self.user_final_predicted_score = {}

        self.metrics = {}
        self.metrics_name = ['precision', 'recall', 'pr', 'ild', 'gc', 'epc']
        self.except_final_rec = except_final_rec
        self.welcome_message()
        self.CHKS = 50 # chunk size for process pool executor
        self.CHKSL = 100# chunk size for process pool executor largest
        self.cache = defaultdict(dict)

    def message_start_section(self,string):
        print("------===%s===------" % (string))
    
    def welcome_message(self):
        #print("Chunk size is %d" % (self.CHKS))
        pass
    @property
    def data_directory(self):
        return self._data_directory

    @data_directory.setter
    def data_directory(self, data_directory):
        if data_directory[-1] != '/':
            data_directory += '/'
        print(f"Setting data directory to {data_directory}")
        self._data_directory = data_directory

    @property
    def base_rec(self):
        return self._base_rec

    @base_rec.setter
    def base_rec(self, base_rec):
        if base_rec not in self.BASE_RECOMMENDERS:
            self._base_rec = next(iter(self.BASE_RECOMMENDERS))
            print(f"Base recommender not detected, using default:{self._base_rec}")
        else:
            self._base_rec = base_rec
        # parametros para o metodo base
        self.base_rec_parameters = {}

    @property
    def final_rec(self):
        return self._final_rec

    @final_rec.setter
    def final_rec(self, final_rec):
        if final_rec not in self.FINAL_RECOMMENDERS:
            self._final_rec = next(iter(self.FINAL_RECOMMENDERS))
            print(f"Base recommender not detected, using default:{self._final_rec}")
        else:
            self._final_rec = final_rec
        # parametros para o metodo final
        self.final_rec_parameters = {}

    @property
    def final_rec_parameters(self):
        return self._final_rec_parameters

    @final_rec_parameters.setter
    def final_rec_parameters(self, parameters):
        final_parameters = self.get_final_parameters()[self.final_rec]
        # for parameter in parameters.copy():
        #     if parameter not in final_parameters:
        #         del parameters[parameter]
        parameters_result = dict()
        for parameter in final_parameters:
            if parameter not in parameters:
                parameters_result[parameter] = final_parameters[parameter]
            else:
                parameters_result[parameter] = parameters[parameter]
        self._final_rec_parameters = parameters_result

    @property
    def base_rec_parameters(self):
        return self._base_rec_parameters

    @base_rec_parameters.setter
    def base_rec_parameters(self, parameters):
        base_parameters = self.get_base_parameters()[self.base_rec]
        # for parameter in parameters.copy():
        #     if parameter not in base_parameters:
        #         del parameters[parameter]
        parameters_result = dict()
        for parameter in base_parameters:
            if parameter not in parameters:
                parameters_result[parameter] = base_parameters[parameter]
            else:
                parameters_result[parameter] = parameters[parameter]
        self._base_rec_parameters = parameters_result
    
    @staticmethod
    def get_base_parameters():
        return {
            "mostpopular": {},
            "usg": {'alpha': 0.1, 'beta': 0.1, 'eta': 0.05}
        }

    @staticmethod
    def get_final_parameters():
        return  {
            "geocat": {'div_weight':0.75,'div_geo_cat_weight':0.75, 'heuristic': 'local_max'},
            "persongeocat": {'div_weight':0.75,'cat_div_method':'ld'},
            "geodiv": {'div_weight':0.75},
            "ld": {'div_weight':0.75},
            "binomial": {'alpha': 0.5, 'div_weight': 0.75}
        }

    def get_base_rec_name(self):
        list_parameters=list(map(str,dict_to_list(self.base_rec_parameters)))
        string="_" if len(list_parameters)>0 else ""
        return f"{self.city}_{self.base_rec}_{self.base_rec_list_size}"+\
                              string+'_'.join(list_parameters)

    def get_final_rec_name(self):
        list_parameters=list(map(str,dict_to_list(self.final_rec_parameters)))
        string="_" if len(list_parameters)>0 else ""
        return f"{self.get_base_rec_name()}_{self.final_rec}_{self.final_rec_list_size}"+\
            string+'_'.join(list_parameters)
    def get_base_rec_short_name(self):
        #list_parameters=list(map(str,dict_to_list(self.base_rec_parameters)))
        if self.base_rec == "mostpopular":
            return self.base_rec
        elif self.base_rec == "usg":
            return self.base_rec
        else:
            return self.base_rec
        #string="_" if len(list_parameters)>0 else ""
    def get_final_rec_short_name(self):
        if self.final_rec == 'geocat':
            return f"{self.get_base_rec_short_name()}_{self.final_rec}_{self.final_rec_parameters['heuristic']}"
        elif self.final_rec == 'persongeocat':
            return f"{self.get_base_rec_short_name()}_{self.final_rec}_{self.final_rec_parameters['cat_div_method']}"
        elif self.final_rec == 'geodiv':
            return f"{self.get_base_rec_short_name()}_{self.final_rec}"
        else:
            return f"{self.get_base_rec_short_name()}_{self.final_rec}"

    def get_base_rec_file_name(self):
        return self.get_base_rec_name()+".json"

    def get_final_rec_file_name(self):
        return self.get_final_rec_name()+".json"

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
        self.user_num=user_num
        self.poi_num=poi_num
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
        self.test_data()
        self.CHKS = int(len(self.all_uids)/multiprocessing.cpu_count()/2)
        self.CHKSL = int(len(self.all_uids)/multiprocessing.cpu_count())
        self.welcome_load()

    def welcome_load(self):
        self.message_start_section("LOAD FINAL MESSAGE")
        print("Chunk size set to %d for this base" %(self.CHKSL))

    def not_in_ground_truth_message(uid):
        print(f"{uid} not in ground_truth [ERROR]")


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

        self.cache[self.base_rec]['U'] = U
        self.cache[self.base_rec]['S'] = S
        self.cache[self.base_rec]['G'] = G

        print("Running usg")
        args=[(uid, alpha, beta) for uid in self.all_uids]
        print("args memory usage:",asizeof.asizeof(args)/1024**2,"MB")
        print("U memory usage:",asizeof.asizeof(U)/1024**2,"MB")
        print("S memory usage:",asizeof.asizeof(S)/1024**2,"MB")
        print("G memory usage:",asizeof.asizeof(G)/1024**2,"MB")
        results = run_parallel(self.run_usg,args,self.CHKS)
        print("usg terminated")

        self.save_result(results,base=True)

    def mostpopular(self):
        args=[(uid,) for uid in self.all_uids]
        #results = run_parallel(self.run_mostpopular,args,self.CHKS)
        results = run_parallel(self.run_mostpopular,args,self.CHKS)
        self.save_result(results,base=True)

    def geocat(self):
        # results = []
        # for uid in self.all_uids:
        #     results.append(self.run_geocat(uid))
        args=[(uid,) for uid in self.all_uids]
        results = run_parallel(self.run_geocat,args,self.CHKS)
        self.save_result(results,base=False)

    def persongeocat(self):
        print("Computing geographic diversification propensity")
        pgeo_div_runner = GeoDivPropensity(self.training_matrix, self.poi_coos)
        self.geo_div_propensity = pgeo_div_runner.geo_div_walk()
        
        pcat_div_runner = CatDivPropensity(
            self.training_matrix,
            cat_utils.get_users_cat_visits(self.training_matrix,
                                           self.poi_cats),
            self.undirected_category_tree,
            cat_div_method=self.final_rec_parameters['cat_div_method'])
        print("Computing categoric diversification propensity with",
              self.final_rec_parameters['cat_div_method'])
        self.cat_div_propensity=pcat_div_runner.compute_cat_div_propensity()
       #print(self.cat_div_propensity)
        # self.beta=geo_div_propensity/np.add(geo_div_propensity,cat_div_propensity)
        args=[(uid,) for uid in self.all_uids]
        results = run_parallel(self.run_persongeocat,args,self.CHKS)
        self.save_result(results,base=False)

    def geodiv(self):
        args=[(uid,) for uid in self.all_uids]
        results = run_parallel(self.run_geodiv,args,self.CHKS)
        self.save_result(results,base=False)

    def ld(self):
        args=[(uid,) for uid in self.all_uids]
        results = run_parallel(self.run_ld,args,self.CHKS)
        self.save_result(results,base=False)

    def binomial(self):
        self.binomial=Binomial(self.training_matrix,self.poi_cats,
            self.final_rec_parameters['div_weight'],self.final_rec_parameters['alpha'])
        self.binomial.compute_all_probabilities()
        # predicted = self.user_base_predicted_lid[0][
        #     0:self.base_rec_list_size]
        # overall_scores = self.user_base_predicted_score[0][
        #     0:self.base_rec_list_size]
        # self.binomial.binomial(0,predicted,overall_scores,self.final_rec_list_size)
        args=[(uid,) for uid in self.all_uids]
        results = run_parallel(self.run_binomial,args,self.CHKS)
        self.save_result(results,base=False)

    @classmethod
    def run_usg(cls, uid, alpha, beta):
        self = cls.getInstance()
        U = self.cache[self.base_rec]['U']
        S = self.cache[self.base_rec]['S']
        G = self.cache[self.base_rec]['G']
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
            # print(uid)
            return json.dumps({'user_id': uid, 'predicted': list(map(int, predicted)), 'score': list(map(float, overall_scores))})+"\n"
        self.not_in_ground_truth_message()
        return ""
    @classmethod
    def run_mostpopular(cls,uid):
        self = cls.getInstance()
        if uid in self.ground_truth:
            poi_indexes = set(list(range(self.poi_num)))
            visited_indexes = set(self.training_matrix[uid].nonzero()[0])
            not_visited_indexes = poi_indexes-visited_indexes
            not_visited_indexes = np.array(list(not_visited_indexes))
            poi_visits_nu = np.count_nonzero(self.training_matrix, axis=0)
            pois_score = poi_visits_nu/self.user_num
            for i in visited_indexes:
                pois_score[i] = 0
            predicted = list(reversed(np.argsort(pois_score)))[
                0:self.base_rec_list_size]
            overall_scores = list(reversed(np.sort(pois_score)))[
                0:self.base_rec_list_size]

            self.user_base_predicted_lid[uid]=predicted
            self.user_base_predicted_score[uid]=overall_scores
            return json.dumps({'user_id': uid, 'predicted': list(map(int, predicted)), 'score': list(map(float, overall_scores))})+"\n"
        self.not_in_ground_truth_message()
        return ""

    @classmethod
    def run_geocat(cls, uid):
        self = cls.getInstance()
        if uid in self.ground_truth:
            predicted = self.user_base_predicted_lid[uid][
                0:self.base_rec_list_size]
            overall_scores = self.user_base_predicted_score[uid][
                0:self.base_rec_list_size]

            # start_time = time.time()

            predicted, overall_scores = gcobjfunc.geocat(uid, self.training_matrix, predicted, overall_scores,
                                                         self.poi_cats, self.poi_neighbors, self.final_rec_list_size, self.undirected_category_tree,
                                                         self.final_rec_parameters['div_geo_cat_weight'],self.final_rec_parameters['div_weight'],
                                                         self.final_rec_parameters['heuristic'])

            # print("uid → %d, time → %fs" % (uid, time.time()-start_time))
            predicted = np.array(predicted)[list(
                reversed(np.argsort(overall_scores)))]
            overall_scores = list(reversed(np.sort(overall_scores)))

            return json.dumps({'user_id': uid, 'predicted': list(map(int, predicted)), 'score': list(map(float, overall_scores))})+"\n"
        self.not_in_ground_truth_message()
        return ""

    @classmethod
    def run_persongeocat(cls,uid):
        self = cls.getInstance()
        if uid in self.ground_truth:
            predicted = self.user_base_predicted_lid[uid][
                0:self.base_rec_list_size]
            overall_scores = self.user_base_predicted_score[uid][
                0:self.base_rec_list_size]
            
            # start_time = time.time()
            if self.geo_div_propensity[uid] == 0 and self.cat_div_propensity[uid] == 0:
                predicted, overall_scores = gcobjfunc.geocat(uid, self.training_matrix, predicted, overall_scores,
                                                self.poi_cats, self.poi_neighbors, self.final_rec_list_size, self.undirected_category_tree,
                                                             0,0,
                                                             'local_max',
                                                             gcobjfunc.persongeocat_objective_function)
            else:
                div_geo_cat_weight=(self.geo_div_propensity[uid])/(self.geo_div_propensity[uid]+self.cat_div_propensity[uid])
                predicted, overall_scores = gcobjfunc.geocat(uid, self.training_matrix, predicted, overall_scores,
                                                         self.poi_cats, self.poi_neighbors, self.final_rec_list_size, self.undirected_category_tree,
                                                             div_geo_cat_weight,self.final_rec_parameters['div_weight'],
                                                             'local_max',
                                                             gcobjfunc.persongeocat_objective_function)

            # print("uid → %d, time → %fs" % (uid, time.time()-start_time))

            predicted = np.array(predicted)[list(
                reversed(np.argsort(overall_scores)))]
            overall_scores = list(reversed(np.sort(overall_scores)))

            return json.dumps({'user_id': uid, 'predicted': list(map(int, predicted)), 'score': list(map(float, overall_scores))})+"\n"
        self.not_in_ground_truth_message()
        return ""

    @classmethod
    def run_geodiv(cls, uid):
        self = cls.getInstance()
        if uid in self.ground_truth:
            predicted = self.user_base_predicted_lid[uid][
                0:self.base_rec_list_size]
            overall_scores = self.user_base_predicted_score[uid][
                0:self.base_rec_list_size]
            
            # start_time = time.time()

            predicted, overall_scores = gcobjfunc.geodiv(uid, self.training_matrix, predicted, overall_scores,
                                                         self.poi_neighbors, self.final_rec_list_size,
                                                         self.final_rec_parameters['div_weight'])

            # print("uid → %d, time → %fs" % (uid, time.time()-start_time))

            predicted = np.array(predicted)[list(
                reversed(np.argsort(overall_scores)))]
            overall_scores = list(reversed(np.sort(overall_scores)))

            return json.dumps({'user_id': uid, 'predicted': list(map(int, predicted)), 'score': list(map(float, overall_scores))})+"\n"
        self.not_in_ground_truth_message()
        return ""

    @classmethod
    def run_ld(cls, uid):
        self = cls.getInstance()
        if uid in self.ground_truth:
            predicted = self.user_base_predicted_lid[uid][
                0:self.base_rec_list_size]
            overall_scores = self.user_base_predicted_score[uid][
                0:self.base_rec_list_size]
            
            predicted, overall_scores = gcobjfunc.ld(uid, self.training_matrix, predicted, overall_scores,
                                                self.poi_cats,self.undirected_category_tree,self.final_rec_list_size,
                                                self.final_rec_parameters['div_weight'])

            predicted = np.array(predicted)[list(
                reversed(np.argsort(overall_scores)))]
            overall_scores = list(reversed(np.sort(overall_scores)))

            return json.dumps({'user_id': uid, 'predicted': list(map(int, predicted)), 'score': list(map(float, overall_scores))})+"\n"
        self.not_in_ground_truth_message()
        return ""
    
    @classmethod
    def run_binomial(cls,uid):
        self = cls.getInstance()
        if uid in self.ground_truth:
            predicted = self.user_base_predicted_lid[uid][
                0:self.base_rec_list_size]
            overall_scores = self.user_base_predicted_score[uid][
                0:self.base_rec_list_size]
            predicted, overall_scores=self.binomial.binomial(uid,predicted,overall_scores,self.final_rec_list_size)
            # predicted, overall_scores = gcobjfunc.ld(uid, self.training_matrix, predicted, overall_scores,
            #                                     self.poi_cats,self.undirected_category_tree,self.final_rec_list_size,
            #                                     self.final_rec_parameters['div_weight'])

            predicted = np.array(predicted)[list(
                reversed(np.argsort(overall_scores)))]
            overall_scores = list(reversed(np.sort(overall_scores)))

            return json.dumps({'user_id': uid, 'predicted': list(map(int, predicted)), 'score': list(map(float, overall_scores))})+"\n"
        self.not_in_ground_truth_message()
        return ""

    def load_base_predicted(self):
        result_file = open(self.data_directory+"result/reclist/"+self.get_base_rec_file_name(), 'r')
        for i,line in enumerate(result_file):
            obj=json.loads(line)
            self.user_base_predicted_lid[obj['user_id']]=obj['predicted']
            self.user_base_predicted_score[obj['user_id']]=obj['score']

    def load_final_predicted(self):
        result_file = open(self.data_directory+"result/reclist/"+self.get_final_rec_file_name(), 'r')

        for i,line in enumerate(result_file):
            obj=json.loads(line)
            self.user_final_predicted_lid[obj['user_id']]=obj['predicted']
            self.user_final_predicted_score[obj['user_id']]=obj['score']

    def message_recommender(self,base):
        if base:
            print(f"{self.base_rec} base recommender")
        else:
            print(f"{self.final_rec} final recommender")
        self.print_parameters(base=base)

    def run_base_recommender(self):
        base_recommender=self.BASE_RECOMMENDERS[self.base_rec]
        self.message_recommender(base=True)
        base_recommender()

    def run_final_recommender(self):
        final_recommender=self.FINAL_RECOMMENDERS[self.final_rec]
        if len(self.user_base_predicted_lid)>0:
            self.message_recommender(base=False)
            final_recommender()
        else:
            print("User base predicted list is empty")
        pass

    def run_all_base(self):
        for recommender in self.BASE_RECOMMENDERS:
            self.base_rec=recommender
            print(f"Running {recommender}")
            self.run_base_recommender()
    def run_all_final(self):
        print(f"Running all final recommenders, base recommender is {self.base_rec}")
        for recommender in self.FINAL_RECOMMENDERS:
            if recommender not in self.except_final_rec:
                self.final_rec = recommender
                print(f"Running {recommender}")
                self.run_final_recommender()
    def run_all_eval(self):
        print(f"Evaluating all final recommenders, base recommender is {self.base_rec}")
        for recommender in self.FINAL_RECOMMENDERS:
            try:
                self.final_rec = recommender
                print(f"Running {recommender}")
                self.load_final_predicted()
            except Exception as e:
                print(e)
                print(f"Maybe recommender {recommender} doesnt have final predicted")
                continue
            self.eval_rec_metrics()

    def eval(self,uid,base,k):
        if base:
            predictions = self.user_base_predicted_lid
            predicted=self.user_base_predicted_lid[uid]
        else:
            predictions = self.user_final_predicted_lid
            predicted=self.user_final_predicted_lid[uid]
        actual=self.ground_truth[uid]
        
        predicted_at_k=predicted[:k]
        precision_val=metrics.precisionk(actual, predicted_at_k)
        rec_val=metrics.recallk(actual, predicted_at_k)
        pr_val=metrics.prk(self.training_matrix[uid],predicted_at_k,self.poi_neighbors)
        ild_val=metrics.ildk(predicted_at_k,self.poi_cats,self.undirected_category_tree)
        gc_val=metrics.gck(uid,self.training_matrix,self.poi_cats,predicted_at_k)
        # this epc is made based on some article of kdd'13
        # A New Collaborative Filtering Approach for Increasing the Aggregate Diversity of Recommender Systems
        # Not the most liked but the used in original geocat
        # if not hasattr(self,'epc_val'):
        # epc_val=metrics.old_global_epck(self.training_matrix,self.ground_truth,predictions,self.all_uids)
        epc_val=self.epc_val
        # else:
        #     epc_val=self.epc_val
        # if uid == max(self.all_uids):
        #     del self.epc_val
        # this epc is maded like vargas, recsys'11
        #epc_val=metrics.epck(predicted_at_k,actual,uid,self.training_matrix)
        
        d={'user_id':uid,'precision':precision_val,'recall':rec_val,'pr':pr_val,'ild':ild_val,'gc':gc_val,'epc':epc_val}

        return json.dumps(d)+'\n'
    def eval_rec_metrics(self,*,base=False):
        if base:
            predictions = self.user_base_predicted_lid
        else:
            predictions = self.user_final_predicted_lid

        for i,k in enumerate(experiment_constants.METRICS_K):
            print(f"running metrics at @{k}")
            self.epc_val = metrics.old_global_epck(self.training_matrix,self.ground_truth,predictions,self.all_uids)

            if base:
                result_out = open(self.data_directory+"result/metrics/"+self.get_base_rec_name()+f"_{str(k)}{R_FORMAT}", 'w')
            else:
                result_out = open(self.data_directory+"result/metrics/"+self.get_final_rec_name()+f"_{str(k)}{R_FORMAT}", 'w')
            
            self.message_recommender(base=base)

            args=[(uid,base,k) for uid in self.all_uids]
            results = run_parallel(self.eval,args,self.CHKSL)
            
            for json_string_result in results:
                result_out.write(json_string_result)
            result_out.close()

    def load_metrics(self,*,base=False,short_name=True):
        if base:
            rec_using=self.base_rec
            if short_name:
                rec_short_name=self.get_base_rec_short_name()
            else:
                rec_short_name=self.get_base_rec_name()
        else:
            rec_using=self.final_rec
            if short_name:
                rec_short_name=self.get_final_rec_short_name()
            else:
                rec_short_name=self.get_final_rec_name()

        self.metrics[rec_short_name]={}
        for i,k in enumerate(experiment_constants.METRICS_K):
            if base:
                result_file = open(self.data_directory+"result/metrics/"+self.get_base_rec_name()+f"_{str(k)}{R_FORMAT}", 'r')
            else:
                result_file = open(self.data_directory+"result/metrics/"+self.get_final_rec_name()+f"_{str(k)}{R_FORMAT}", 'r')
            
            self.metrics[rec_short_name][k]=[]
            for i,line in enumerate(result_file):
                obj=json.loads(line)
                self.metrics[rec_short_name][k].append(obj)




    def plot_bar_metrics(self):
        palette = plt.get_cmap('Set1')
        for i,k in enumerate(experiment_constants.METRICS_K):

            
            metrics_mean=dict()
            for i,rec_using,metrics in zip(range(len(self.metrics)),self.metrics.keys(),self.metrics.values()):
                metrics=metrics[k]
                
                metrics_mean[rec_using]=defaultdict(float)
                for obj in metrics:
                    for key in obj:
                        if key in self.metrics_name:
                            metrics_mean[rec_using][key]+=obj[key]
                
                
                for j,key in enumerate(metrics_mean[rec_using]):
                    metrics_mean[rec_using][key]/=len(metrics)
                    #print(f"{key}:{metrics_mean[rec_using][key]}")
            fig = plt.figure()
            ax=fig.add_subplot(111)
            barWidth= 1-len(self.metrics)/(1+len(self.metrics))
            N=len(self.metrics_name)
            indexes=np.arange(N)
            i=0
            for rec_using,rec_metrics in metrics_mean.items():
                print(f"{rec_using} at @{k}")
                print(rec_metrics.values())
                ax.bar(indexes+i*barWidth,rec_metrics.values(),barWidth,label=rec_using,color=palette(i))
                #ax.bar(indexes[j]+i*barWidth,np.mean(list(rec_metrics.values())),barWidth,label=rec_using,color=palette(i))
                i+=1
            metrics_utility_score=dict()
            for rec_using in metrics_mean:
                metrics_utility_score[rec_using]=dict()
            for metric_name in self.metrics_name:
                max_metric_value=0
                min_metric_value=1
                for rec_using,rec_metrics in metrics_mean.items():
                    max_metric_value=max(max_metric_value,rec_metrics[metric_name])
                    min_metric_value=min(min_metric_value,rec_metrics[metric_name])
                for rec_using,rec_metrics in metrics_mean.items():
                    metrics_utility_score[rec_using][metric_name]=(rec_metrics[metric_name]-min_metric_value)\
                        /(max_metric_value-min_metric_value)
            i=0
            for rec_using,utility_scores in metrics_utility_score.items():
                ax.bar(N+i*barWidth,np.sum(list(utility_scores.values()))/len(utility_scores),barWidth,label=rec_using,color=palette(i))
                i+=1
            
                

            #ax.set_xticks(np.arange(N+1)+barWidth*(np.floor((len(self.metrics))/2)-1)+barWidth/2)
            ax.set_xticks(np.arange(N+1)+barWidth*(((len(self.metrics))/2)-1)+barWidth/2)
            # ax.legend((p1[0], p2[0]), self.metrics_name)
            ax.legend(tuple(self.metrics.keys()))
            ax.set_xticklabels(self.metrics_name+['MAUT'])
            ax.set_title(f"at @{k}, {self.city}")
            fig.show()
            plt.show()
            timestamp = datetime.timestamp(datetime.now())
            fig.savefig(self.data_directory+f"result/img/all_met_{self.city}_{str(k)}_{timestamp}.png")
            
                # ax.bar(indexes[j+1]+i*barWidth,np.mean(list(metrics_mean[rec_using].values())),barWidth,label=rec_using,color=palette(i))
    def test_data(self):
        self.message_start_section("TESTING DATA SET")
        has_some_error_global = False
        for i in self.all_uids:
            has_some_error = False
            test_size = len(self.ground_truth[i])
            train_size = np.count_nonzero(self.training_matrix[i])
            if test_size == 0:
                print(f"user {i} with empty ground truth")
                has_some_error = True
                # remove from tests
                self.all_uids.remove(i)
            if train_size == 0:
                print(f"user {i} with empty training data!!!!! Really bad error")
                has_some_error = True
            if has_some_error:
                has_some_error_global = True
                print("Training size is %d, test size is %d" %\
                      (train_size,test_size))
        if not has_some_error_global:
            print("No error encountered in base")
            
    def print_parameters(self, base):
        print_dict(self.base_rec_parameters)
        if base == False:
            print()
            print_dict(self.final_rec_parameters)


    def plot_geocat_parameters_metrics(self):

        for i,K in enumerate(experiment_constants.METRICS_K):
            palette = plt.get_cmap('Set1')
            fig = plt.figure(figsize=(8,8))
            ax=fig.add_subplot(111)
            plt.xticks(rotation='vertical')
            #K = max(experiment_constants.METRICS_K)
            #K = 10
            metrics_mean=dict()
            x = 1
            r = 0.25
            l = []
            for i in np.append(np.arange(0, x, r),x):
                for j in np.append(np.arange(0, x, r),x):
                    if not(i==0 and i!=j):
                        l.append((i,j))
            self.base_rec = "usg"
            self.final_rec = "geocat"
            rec_using = self.get_final_rec_short_name()
            # There is some ways to do it more efficiently but i could not draw lines between points
            # this way is ultra slow but works
            for i, metric_name in enumerate(self.metrics_name):
                metric_values = []
                for div_weight, div_geo_cat_weight in l:
                    self.final_rec_parameters['div_weight'], self.final_rec_parameters['div_geo_cat_weight'] = div_weight, div_geo_cat_weight
                    self.load_metrics(base=False)
                    metrics=self.metrics[rec_using][K]
                    metrics_mean[rec_using]=defaultdict(float)
                    for obj in metrics:
                        metrics_mean[rec_using][metric_name]+=obj[metric_name]
                    metrics_mean[rec_using][metric_name]/=len(metrics)
                    metric_values.append(metrics_mean[rec_using][metric_name])
                ax.plot(list(map(str,l)),metric_values, '-o',color=palette(i))
            ax.legend(tuple(self.metrics_name))
            fig.show()
            plt.show()
            timestamp = datetime.timestamp(datetime.now())
            fig.savefig(self.data_directory+f"result/img/geocat_parameters_{self.city}_{str(K)}_{timestamp}.png")
