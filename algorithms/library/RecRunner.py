from methods.random import random_diversifier
import ctypes
import utils
from constants import *
import geo_utils
from methods.gc import gc_diversifier
from Text import Text
from geosoca.CategoricalCorrelation import CategoricalCorrelation
from geosoca.SocialCorrelation import SocialCorrelation
from geosoca.AdaptiveKernelDensityEstimation import AdaptiveKernelDensityEstimation
from parallel_util import run_parallel
from Pm2 import Pm2
from Binomial import Binomial
import metrics
from constants import experiment_constants, RECS_PRETTY, CITIES_PRETTY, HEURISTICS_PRETTY, CITIES_BEST_PARAMETERS
from GeoDiv2020 import GeoDiv2020
from GeoMF import GeoMF
from pgc.CatDivPropensity import CatDivPropensity
from pgc.GeoDivPropensity import GeoDivPropensity
import objfunc as gcobjfunc
from usg.PowerLaw import PowerLaw
from usg.FriendBasedCF import FriendBasedCF
from usg.UserBasedCF import UserBasedCF
import cat_utils
from matplotlib.legend import Legend
import scipy.stats
import imblearn.under_sampling
import imblearn.over_sampling
import imblearn.datasets
import imblearn
from sklearn.cluster import DBSCAN
from sklearn import neighbors
from sklearn.preprocessing import PolynomialFeatures
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import scipy
from pympler import asizeof
from cycler import cycler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
from scipy.stats import describe
from tqdm import tqdm
import numpy as np
import inquirer
import math
import os
from collections import Counter
import multiprocessing
import itertools
from datetime import datetime
import time
import json
from concurrent.futures import ProcessPoolExecutor
import pickle
from collections import defaultdict, OrderedDict, Counter
from abc import ABC, abstractmethod
from enum import Enum
LANG = 'en'


class NameType(Enum):
    SHORT = 1
    PRETTY = 2
    BASE_PRETTY = 3
    CITY_BASE_PRETTY = 4
    FULL = 5


def sec_to_hm(s):
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f'{h:d}h{m:02d}m'


MPL_ALPHA = 0.5

CMAP_NAME = 'viridis'

# DATA_DIRECTORY = '../data'  # directory with all data

# CHKS = 40 # chunk size for process pool executor
# CHKSL = 200 # chunk size for process pool executor largest

regressor_predictors = {
    # 'Linear regressor' : LinearRegression(),
    # 'Neural network regressor' : MLPRegressor(hidden_layer_sizes=(20,20,20,20)),
}

classifier_predictors = {
    'RFC': RandomForestClassifier(n_estimators=500),
    # 'MLPC' : MLPClassifier(hidden_layer_sizes=(20,20,20,20,20,20)),
    'SVM': svm.SVC(),
    'KNN': neighbors.KNeighborsClassifier(),
}


def normalize(scores):
    scores = np.array(scores, dtype=np.float128)

    max_score = np.max(scores)
    if not max_score == 0:
        scores = [s / max_score for s in scores]
    return scores


def dict_to_list_gen(d):
    for k, v in zip(d.keys(), d.values()):
        if v == None:
            continue
        yield k
        yield v


def dict_to_list(d):
    return list(dict_to_list_gen(d))


def print_dict(dictionary):
    for key, value in dictionary.items():
        print(f"{key} : {value}")


class RecRunner():
    PARAMETERS_BY_CITY = True

    def save_result(self, results, base=True):
        if base:
            utils.create_path_to_file(
                self.data_directory+RECLIST + self.get_base_rec_file_name())
            result_out = open(self.data_directory+RECLIST +
                              self.get_base_rec_file_name(), 'w')
        else:
            utils.create_path_to_file(
                self.data_directory+RECLIST + self.get_final_rec_file_name())
            result_out = open(self.data_directory+RECLIST +
                              self.get_final_rec_file_name(), 'w')
        for json_string_result in results:
            result_out.write(json_string_result)
        result_out.close()

    def __init__(self, base_rec, final_rec, city,
                 base_rec_list_size, final_rec_list_size, data_directory,
                 base_rec_parameters={}, final_rec_parameters={}, except_final_rec=[]):
        self.BASE_RECOMMENDERS = {
            "geomf": self.geomf,
            "mostpopular": self.mostpopular,
            "usg": self.usg,
            "geosoca": self.geosoca,
        }
        self.FINAL_RECOMMENDERS = {
            "geodiv2020": self.geodiv2020,
            "geocat": self.geocat,
            "persongeocat": self.persongeocat,
            "geodiv": self.geodiv,
            "ld": self.ld,
            "binomial": self.binomial,
            "pm2": self.pm2,
            "pdpgeocat": self.pdpgeocat,
            "gc": self.gc,
            "random": self.random,
        }
        self.city = city
        self.base_rec = base_rec
        self.final_rec = final_rec

        self.base_rec_parameters = base_rec_parameters
        self.final_rec_parameters = final_rec_parameters

        self.base_rec_list_size = base_rec_list_size
        self.final_rec_list_size = final_rec_list_size

        self.data_directory = data_directory

        self.user_base_predicted_lid = {}
        self.user_base_predicted_score = {}
        self.user_final_predicted_lid = {}
        self.user_final_predicted_score = {}

        self.metrics = {}
        self.groups_epc = {}
        self.metrics_name = ['precision', 'recall', 'gc', 'ild', 'pr', 'epc']

        self.except_final_rec = except_final_rec
        self.welcome_message()
        self.CHKS = 50  # chunk size for process pool executor
        self.CHKSL = 100  # chunk size for process pool executor largest
        self.cache = defaultdict(dict)
        self.metrics_cities = dict()

        self.show_heuristic = False
        self.persons_plot_special_case = False
        self.train_size = None
        self.recs_user_final_predicted_lid = {}
        self.recs_user_final_predicted_score = {}
        self.recs_user_base_predicted_lid = {}
        self.recs_user_base_predicted_score = {}

    def message_start_section(self, string):
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
            print(
                f"Base recommender not detected, using default:{self._base_rec}")
        else:
            self._base_rec = base_rec
        self.base_rec_parameters = {}

    @property
    def final_rec(self):
        return self._final_rec

    @final_rec.setter
    def final_rec(self, final_rec):
        if final_rec not in self.FINAL_RECOMMENDERS:
            # self._final_rec = next(iter(self.FINAL_RECOMMENDERS))
            self._final_rec= None
            # print(
                # f"Base recommender not detected, using default:{self._final_rec}")
        else:
            self._final_rec = final_rec
        self.final_rec_parameters = {}

    @property
    def final_rec_parameters(self):
        return self._final_rec_parameters

    @final_rec_parameters.setter
    def final_rec_parameters(self, parameters):
        if self.PARAMETERS_BY_CITY:
            if self.final_rec != None:
                final_parameters = CITIES_BEST_PARAMETERS[self.base_rec][self.city][self.final_rec]
            else:
                final_parameters = {}
        else:
            final_parameters = self.get_final_parameters()[self.final_rec]
        parameters_result = dict()
        for parameter in final_parameters:
            if parameter not in parameters:
                # default value
                source_of_parameters_value = final_parameters
            else:
                # input value
                source_of_parameters_value = parameters
            parameters_result[parameter] = source_of_parameters_value[parameter]
        # filtering special cases

        if self.final_rec == 'geocat' and parameters_result['heuristic'] == 'local_max' and parameters_result['obj_func'] == None:
            parameters_result['obj_func'] = 'cat_weight'

        if self.final_rec == 'geocat' and parameters_result['heuristic'] == 'local_max' and parameters_result['obj_func'] != 'cat_weight':
            parameters_result['div_cat_weight'] = None
        # if self.final_rec == 'geocat' and parameters_result['obj_func'] == 'cat_weight' and parameters_result['div_cat_weight'] == None:
        #     print("Auto setting div_cat_weight to 0.95")
        #     parameters_result['div_cat_weight'] = 0.95
        if self.final_rec == 'geocat' and parameters_result['heuristic'] != 'local_max':
            print("Auto setting obj_func to None")
            parameters_result['obj_func'] = None

        self._final_rec_parameters = parameters_result

    @property
    def base_rec_parameters(self):
        return self._base_rec_parameters

    @base_rec_parameters.setter
    def base_rec_parameters(self, parameters):
        base_parameters = self.get_base_parameters()[self.base_rec]
        parameters_result = dict()
        for parameter in base_parameters:
            if parameter not in parameters:
                parameters_result[parameter] = base_parameters[parameter]
            else:
                parameters_result[parameter] = parameters[parameter]
        self._base_rec_parameters = parameters_result

    @staticmethod
    def get_base_parameters_descriptions():
        return {
            "geomf": {'K': 'number of latent factors', 'delta': 'distance threshold', 'gamma': 'regularization coefficient', 'epsilon': 'l_1 regularization coefficient', 'lambda_': 'l_1 regularization coefficient', 'max_iters': 'max number of iterations', 'grid_distance': 'grid distance or area width and height'},
            "mostpopular": {},
            "usg": {'alpha': 'social influence', 'beta': 'geographical influence', 'eta': 'social influence weight between social connections and check-ins'},
            "geosoca": {'alpha': 'sensitivity parameter to local bandwith'},
        }


    @staticmethod
    def get_base_parameters():
        return {
            "geomf": {'K': 100, 'delta': 50, 'gamma': 0.01, 'epsilon': 10, 'lambda_': 10, 'max_iters': 7, 'grid_distance': 3.0},
            "mostpopular": {},
            "usg": {'alpha': 0, 'beta': 0.2, 'eta': 0},
            "geosoca": {'alpha': 0.3},
        }

    @classmethod
    def get_final_parameters(cls):
        return {
            "geodiv2020": {'div_weight': 0.5},
            "geocat": {'div_weight': 0.75, 'div_geo_cat_weight': 0.25, 'heuristic': 'local_max', 'obj_func': 'cat_weight', 'div_cat_weight': 0.05},
            "persongeocat": {'div_weight': 0.75, 'cat_div_method': 'inv_num_cat', 'geo_div_method': 'walk',
                             'obj_func': 'cat_weight', 'div_cat_weight': 0.05, 'bins': None,
                             'norm_method': 'default', 'funnel': None},
            "geodiv": {'div_weight': 0.5},
            "ld": {'div_weight': 0.25},
            "binomial": {'alpha': 0.5, 'div_weight': 0.75},
            "pm2": {'div_weight': 1},
            "pdpgeocat": {'k': 10, 'interval': 2, 'div_geo_cat_weight': 0.75},
            "gc": {'div_weight': 0.8},
            "random": {'div_weight': 1},
        }

    def get_base_rec_name(self):
        list_parameters = list(
            map(str, dict_to_list(self.base_rec_parameters)))
        string = "_" if len(list_parameters) > 0 else ""

        k_fold = ''

        if self.train_size != None:
            train_size = f'{self.train_size}_'
        else:
            train_size = ''

        return f"{train_size}{k_fold}{self.city}_{self.base_rec}_{self.base_rec_list_size}" +\
            string+'_'.join(list_parameters)

    def get_final_rec_name(self):
        list_parameters = list(
            map(str, dict_to_list(self.final_rec_parameters)))
        string = "_" if len(list_parameters) > 0 else ""
        return f"{self.get_base_rec_name()}_{self.final_rec}_{self.final_rec_list_size}" +\
            string+'_'.join(list_parameters)

    def get_base_rec_short_name(self):
        if self.base_rec == "mostpopular":
            return self.base_rec
        elif self.base_rec == "usg":
            return self.base_rec
        else:
            return self.base_rec

    def get_final_rec_short_name(self):
        if self.final_rec == 'geocat':
            return f"{self.get_base_rec_short_name()}_{self.final_rec}_{self.final_rec_parameters['div_geo_cat_weight']}_{self.final_rec_parameters['div_cat_weight']}"
        elif self.final_rec == 'persongeocat':
            string = f"{self.get_base_rec_short_name()}_{self.final_rec}"
            if self.persons_plot_special_case:
                string = ''
            string += f"_{self.final_rec_parameters['cat_div_method']}"
            if self.persons_plot_special_case:
                string = string[1:]
            string += f"_{self.final_rec_parameters['geo_div_method']}"
            print(string)
            return string
        elif self.final_rec == 'geodiv':
            return f"{self.get_base_rec_short_name()}_{self.final_rec}"
        elif self.final_rec == 'binomial':
            string = f"{self.get_base_rec_short_name()}_{self.final_rec}"
            string += f"_{self.final_rec_parameters['div_weight']}"
            string += f"_{self.final_rec_parameters['alpha']}"
            return string
        else:
            return f"{self.get_base_rec_short_name()}_{self.final_rec}"

    def get_base_rec_pretty_name(self):
        return RECS_PRETTY[self.base_rec]

    def get_final_rec_pretty_name(self):
        if self.show_heuristic and self.final_rec == 'geocat':
            return f'{RECS_PRETTY[self.final_rec]}({HEURISTICS_PRETTY[self.final_rec_parameters["heuristic"]]})'
        if self.final_rec == 'geocat' and self.final_rec_parameters['div_geo_cat_weight'] == 1:
            if self.final_rec_parameters['div_cat_weight'] == 0:
                return f'{RECS_PRETTY[self.final_rec]} (w.r.t. GC)'
            elif self.final_rec_parameters['div_cat_weight'] == 1:
                return f'{RECS_PRETTY[self.final_rec]} (w.r.t. LD)'
        return RECS_PRETTY[self.final_rec]

    def get_base_rec_file_name(self):
        return self.get_base_rec_name()+".json"

    def get_final_rec_file_name(self):
        return self.get_final_rec_name()+".json"

    def get_base_metrics_name(self):
        return self.data_directory+METRICS+self.get_base_rec_name()+f"{R_FORMAT}"

    def get_final_metrics_name(self):
        return self.data_directory+METRICS+self.get_final_rec_name()+f"{R_FORMAT}"

    def load_base(self, user_data=False, test_data=True):
        CITY = self.city

        print(f"{CITY} city base loading")

        if self.train_size != None:
            self.training_matrix = pickle.load(open(
                self.data_directory+UTIL+f'train_val_{self.get_train_validation_name()}.pickle', 'rb'))
            self.ground_truth = pickle.load(open(
                self.data_directory+UTIL+f'test_val_{self.get_train_validation_name()}.pickle', 'rb'))

        # Train load

        if self.train_size == None:
            self.data_checkin_train = pickle.load(
                open(self.data_directory+TRAIN+CITY+".pickle", "rb"))

            # Test load
            self.ground_truth = defaultdict(set)
            for checkin in pickle.load(open(self.data_directory+TEST+CITY+".pickle", "rb")):
                self.ground_truth[checkin['user_id']].add(checkin['poi_id'])
            self.ground_truth = dict(self.ground_truth)
        # Pois load
        self.poi_coos = {}
        self.poi_cats = {}
        for poi_id, poi in pickle.load(open(self.data_directory+POI+CITY+".pickle", "rb")).items():
            self.poi_coos[poi_id] = tuple([poi['latitude'], poi['longitude']])
            self.poi_cats[poi_id] = poi['categories']

        # Social relations load
        self.social_relations = defaultdict(list)
        for user_id, friends in pickle.load(open(self.data_directory+USER_FRIEND+CITY+".pickle", "rb")).items():
            self.social_relations[user_id] = friends

        self.social_relations = dict(self.social_relations)
        user_num = len(self.social_relations)
        poi_num = len(self.poi_coos)
        user_num, poi_num
        self.user_num = user_num
        self.poi_num = poi_num
        # Cat Hierarchy load
        self.dict_alias_title, self.category_tree, self.dict_alias_depth = cat_utils.cat_structs_igraph(
            self.data_directory+DATASET_DIRECTORY+"categories.json")
        # self.undirected_category_tree = self.category_tree.to_undirected()
        self.undirected_category_tree = self.category_tree.shortest_paths()

        cats = self.category_tree.vs['name']
        self.cat_num = len(cats)
        cats_to_int = dict()
        for i, cat in enumerate(cats):
            cats_to_int[cat] = i
        self.poi_cats = {poi_id: [cats_to_int[cat] for cat in cats]
                         for poi_id, cats in self.poi_cats.items()}

        # Training matrix create
        if self.train_size == None:
            self.training_matrix = np.zeros((user_num, poi_num))
            for checkin in self.data_checkin_train:
                self.training_matrix[checkin['user_id'],
                                     checkin['poi_id']] += 1

        # poi neighbors load
        self.poi_neighbors = pickle.load(
            open(self.data_directory+NEIGHBOR+CITY+".pickle", "rb"))
        print(f"{CITY} city base loaded")
        self.all_uids = list(range(user_num))
        self.all_lids = list(range(poi_num))

        if user_data:
            self.user_data = pickle.load(
                open(self.data_directory+USER+CITY+'.pickle', 'rb'))
        if test_data:
            self.test_data()

            print("Removing invalid users")

            self.training_matrix = self.training_matrix[self.all_uids]

            uid_to_int = dict()
            self.uid_to_int = uid_to_int
            for i, uid in enumerate(self.all_uids):
                uid_to_int[uid] = i

            for uid in self.invalid_uids:
                del self.social_relations[uid]
                if user_data:
                    del self.user_data[uid]

            self.ground_truth = dict((uid_to_int[key], value) for (
                key, value) in self.ground_truth.items())
            self.social_relations = dict((uid_to_int[key], value) for (
                key, value) in self.social_relations.items())
            if user_data:
                self.user_data = dict((uid_to_int[key], value) for (
                    key, value) in self.user_data.items())

            for uid, i in uid_to_int.items():
                self.social_relations[i] = [uid_to_int[friend_uid] for friend_uid in self.social_relations[i]
                                            if friend_uid in uid_to_int]

            self.all_uids = [uid_to_int[uid] for uid in self.all_uids]
            self.user_num = len(self.all_uids)
            print("Finish removing invalid users")

        if user_data:
            self.user_data = pd.DataFrame(self.user_data).T
            self.user_data['yelping_since'] = self.user_data['yelping_since'].apply(
                lambda date: pd.Timestamp(date).year)
            years = list(range(2004, 2019))
            self.user_data = pd.get_dummies(
                self.user_data, columns=['yelping_since'])
            for year in years:
                if f'yelping_since_{year}' not in self.user_data.columns:
                    self.user_data[f'yelping_since_{year}'] = 0
            print("User data memory usage:", asizeof.asizeof(
                self.user_data)/1024**2, "MB")

        self.CHKS = int(len(self.all_uids)/multiprocessing.cpu_count()/8)
        self.CHKSL = int(len(self.all_uids)/multiprocessing.cpu_count())
        self.welcome_load()

    def welcome_load(self):
        self.message_start_section("LOAD FINAL MESSAGE")
        print('user num: %d, poi num: %d, checkin num: %d' % (
            self.training_matrix.shape[0], self.training_matrix.shape[1], self.training_matrix.sum().sum()))
        print("Chunk size set to %d for this base" % (self.CHKS))
        print("Large chunk size set to %d for this base" % (self.CHKSL))

    def not_in_ground_truth_message(self, uid):
        print(f"{uid} not in ground_truth [ERROR]")

    def usg(self):
        training_matrix = self.training_matrix
        all_uids = self.all_uids
        social_relations = self.social_relations
        poi_coos = self.poi_coos
        top_k = self.base_rec_list_size
        training_matrix = training_matrix.copy()
        training_matrix[training_matrix >= 1] = 1
        alpha = self.base_rec_parameters['alpha']
        beta = self.base_rec_parameters['beta']

        U = UserBasedCF()
        S = FriendBasedCF.getInstance(eta=self.base_rec_parameters['eta'])
        G = PowerLaw()

        U.pre_compute_rec_scores(training_matrix)
        S.compute_friend_sim(social_relations, training_matrix)
        G.fit_distance_distribution(training_matrix, poi_coos)

        self.cache[self.base_rec]['U'] = U
        self.cache[self.base_rec]['S'] = S
        self.cache[self.base_rec]['G'] = G

        print("Running USG")
        args = [(id(self),uid, alpha, beta) for uid in self.all_uids]
        print("args memory usage:", asizeof.asizeof(args)/1024**2, "MB")
        print("(U) User collaborative filtering module memory usage:", asizeof.asizeof(U)/1024**2, "MB")
        print("(S) Friend-based collaborative filtering module memory usage:", asizeof.asizeof(S)/1024**2, "MB")
        print("(G) Geographical influence module memory usage:", asizeof.asizeof(G)/1024**2, "MB")
        results = run_parallel(self.run_usg, args, self.CHKS)
        print("usg terminated")
        self.save_result(results, base=True)

    def mostpopular(self):
        args = [(id(self),uid,) for uid in self.all_uids]
        results = run_parallel(self.run_mostpopular, args, self.CHKS)
        self.save_result(results, base=True)

    def geocat(self):
        args = [(id(self),uid,) for uid in self.all_uids]
        results = run_parallel(self.run_geocat, args, self.CHKS)
        self.save_result(results, base=False)

    def persongeocat_preprocess(self):
        if self.final_rec_parameters['cat_div_method'] in list(classifier_predictors.keys()) and self.final_rec_parameters['geo_div_method'] == None:
            bckp = vars(self).copy()

            df_test = self.generate_general_user_data()

            self.city = 'madison'
            self.load_base()
            self.load_perfect()
            perfect_parameter_train = list(self.perfect_parameter.values())
            df = self.generate_general_user_data()

            for key, val in bckp.items():
                vars(self)[key] = val

            # print(len(self.perfect_parameter))

            X_train, X_test, y_train = df.to_numpy(), df_test.to_numpy(), perfect_parameter_train

            lab_enc = preprocessing.LabelEncoder()

            encoded_y_train = lab_enc.fit_transform(y_train)

            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

            name = self.final_rec_parameters['cat_div_method']
            cp = classifier_predictors[name]
            cp.fit(X_train, encoded_y_train)
            pred_cp = cp.predict(X_test)
            print("-------", name, "-------")
            self.div_geo_cat_weight = lab_enc.inverse_transform(pred_cp)
            self.div_weight = np.ones(
                len(self.div_geo_cat_weight))*self.final_rec_parameters['div_weight']
            return

        if self.final_rec_parameters['geo_div_method'] == 'perfect':
            tmp_final_rec = self.final_rec
            tmp_final_rec_parameters = self.final_rec_parameters
            self.final_rec = 'perfectpgeocat'
            self.load_perfect()

            b = [uid for uid in self.all_uids if uid not in self.perfect_parameter]
            for uid in self.all_uids:
                if uid not in self.perfect_parameter:
                    self.perfect_parameter[uid] = 0.0
            self.geo_div_propensity = []
            for uid in self.all_uids:
                self.geo_div_propensity.append(self.perfect_parameter[uid])
            self.geo_div_propensity = np.array(self.geo_div_propensity)

            self.final_rec = tmp_final_rec
            self.final_rec_parameters = tmp_final_rec_parameters

        elif self.final_rec_parameters['geo_div_method'] != None:
            print("Computing geographic diversification propensity")
            self.pgeo_div_runner = GeoDivPropensity.getInstance(self.training_matrix, self.poi_coos,
                                                                self.poi_cats, self.undirected_category_tree,
                                                                self.final_rec_parameters['geo_div_method'])
            self.geo_div_propensity = self.pgeo_div_runner.compute_div_propensity()

            if self.final_rec_parameters['funnel'] == True:
                self.geo_div_propensity[self.geo_div_propensity >= 0.5] = self.geo_div_propensity[self.geo_div_propensity >=
                                                                                                  0.5]**self.geo_div_propensity[self.geo_div_propensity >= 0.5]
                self.geo_div_propensity[self.geo_div_propensity < 0.5] = self.geo_div_propensity[self.geo_div_propensity < 0.5]**(
                    1+self.geo_div_propensity[self.geo_div_propensity < 0.5])

        if self.final_rec_parameters['cat_div_method'] != None:
            self.pcat_div_runner = CatDivPropensity.getInstance(
                self.training_matrix,
                self.undirected_category_tree,
                self.final_rec_parameters['cat_div_method'],
                self.poi_cats)
            print("Computing categoric diversification propensity with",
                  self.final_rec_parameters['cat_div_method'])
            self.cat_div_propensity = self.pcat_div_runner.compute_div_propensity()
            if self.final_rec_parameters['funnel'] == True:
                self.cat_div_propensity[self.cat_div_propensity >= 0.5] = self.cat_div_propensity[self.cat_div_propensity >=
                                                                                                  0.5]**self.cat_div_propensity[self.cat_div_propensity >= 0.5]
                self.cat_div_propensity[self.cat_div_propensity < 0.5] = self.cat_div_propensity[self.cat_div_propensity < 0.5]**(
                    1+self.cat_div_propensity[self.cat_div_propensity < 0.5])

        if self.final_rec_parameters['norm_method'] == 'default':
            if self.final_rec_parameters['cat_div_method'] == None:
                self.div_geo_cat_weight = 1-self.geo_div_propensity
                self.div_weight = np.ones(
                    len(self.div_geo_cat_weight))*self.final_rec_parameters['div_weight']
            elif self.final_rec_parameters['geo_div_method'] == None:
                self.div_geo_cat_weight = self.cat_div_propensity
                self.div_weight = np.ones(
                    len(self.div_geo_cat_weight))*self.final_rec_parameters['div_weight']
            else:
                self.div_geo_cat_weight = self.cat_div_propensity*self.geo_div_propensity
                self.div_weight = np.ones(
                    len(self.div_geo_cat_weight))*self.final_rec_parameters['div_weight']

        elif self.final_rec_parameters['norm_method'] == 'quadrant':
            cat_div_propensity = self.cat_div_propensity
            geo_div_propensity = self.geo_div_propensity
            cat_median = 0.5
            geo_median = 0.5
            groups = dict()
            groups['geo_preference'] = (cat_div_propensity <= cat_median) & (
                geo_div_propensity > geo_median)
            groups['no_preference'] = ((cat_div_propensity <= cat_median) & (
                geo_div_propensity <= geo_median))
            groups['geocat_preference'] = (
                (cat_div_propensity >= cat_median) & (geo_div_propensity >= geo_median))
            groups['cat_preference'] = (cat_div_propensity > cat_median) & (
                geo_div_propensity <= geo_median)
            self.div_geo_cat_weight = np.zeros(self.training_matrix.shape[0])
            self.div_geo_cat_weight[groups['geo_preference']] = 0
            self.div_geo_cat_weight[groups['geocat_preference']] = 0.5
            self.div_geo_cat_weight[groups['cat_preference']] = 1
            self.div_weight = np.ones(
                len(self.div_geo_cat_weight))*self.final_rec_parameters['div_weight']
            self.div_weight[groups['no_preference']] = 0
            # self.div_geo_cat_weight[groups['no_preference']] = 0.25
            uid_group = dict()
            for group, array in groups.items():
                uid_group.update(dict.fromkeys(np.nonzero(array)[0], group))
            fgroups = open(self.data_directory+UTIL +
                           f'groups_{self.get_final_rec_name()}.pickle', 'wb')
            pickle.dump(uid_group, fgroups)
            fgroups.close()

        if self.final_rec_parameters['bins'] != None:
            bins = np.append(
                np.arange(0, 1, 1/(self.final_rec_parameters['bins']-1)), 1)
            centers = (bins[1:]+bins[:-1])/2
            self.div_geo_cat_weight = bins[np.digitize(
                self.div_geo_cat_weight, centers)]
        # fout = open(self.data_directory+UTIL+f'parameter_{self.get_final_rec_name()}.pickle',"wb")
        # pickle.dump(self.div_geo_cat_weight,fout)
        # fout.close()

        if self.final_rec_parameters['div_cat_weight'] == None:
            self.pcat_div_runner = CatDivPropensity.getInstance(
                self.training_matrix,
                self.undirected_category_tree,
                'qtd_cat',
                self.poi_cats)
            print("Computing categoric diversification propensity with",
                  self.final_rec_parameters['cat_div_method'])
            _cat_div_propensity = self.pcat_div_runner.compute_div_propensity()
            self.div_cat_weight = np.ones(
                len(self.div_geo_cat_weight))*_cat_div_propensity
        else:
            self.div_cat_weight = np.ones(
                len(self.div_geo_cat_weight))*self.final_rec_parameters['div_cat_weight']
        # print(self.div_cat_weight)
        # raise SystemExit

    def persongeocat(self):
        self.persongeocat_preprocess()
        args = [(id(self),uid,) for uid in self.all_uids]
        results = run_parallel(self.run_persongeocat, args, self.CHKS)
        self.save_result(results, base=False)

    def geodiv(self):
        args = [(id(self),uid,) for uid in self.all_uids]
        results = run_parallel(self.run_geodiv, args, self.CHKS)
        self.save_result(results, base=False)

    def ld(self):
        args = [(id(self),uid,) for uid in self.all_uids]
        results = run_parallel(self.run_ld, args, self.CHKS)
        self.save_result(results, base=False)

    def binomial(self):
        self.binomial = Binomial.getInstance(self.training_matrix, self.poi_cats, self.cat_num,
                                             self.final_rec_parameters['div_weight'], self.final_rec_parameters['alpha'])
        self.binomial.compute_all_probabilities()
        args = [(id(self),uid,) for uid in self.all_uids]
        results = run_parallel(self.run_binomial, args, self.CHKS)
        self.save_result(results, base=False)
        del self.binomial

    def pm2(self):
        self.pm2 = Pm2(self.training_matrix, self.poi_cats,
                       self.final_rec_parameters['div_weight'])
        args = [(id(self),uid,) for uid in self.all_uids]
        results = run_parallel(self.run_pm2, args, self.CHKS)
        self.save_result(results, base=False)
        del self.pm2

    def pdpgeocat(self):
        args = [(id(self),uid,) for uid in self.all_uids]
        results = run_parallel(self.run_pdpgeocat, args, self.CHKS)

        uids = [r[1] for r in results]
        div_weights = [r[2] for r in results]
        self.perfect_div_weight = dict()
        for uid, div_weight in zip(uids, div_weights):
            self.perfect_div_weight[uid] = div_weight
        results = [r[0] for r in results]

        fout = open(self.data_directory+UTIL +
                    f'parameter_{self.get_final_rec_name()}.pickle', "wb")
        pickle.dump(self.perfect_div_weight, fout)
        self.save_result(results, base=False)

    def pdpgeocat(self):
        args = [(id(self),uid,) for uid in self.all_uids]
        results = run_parallel(self.run_pdpgeocat, args, self.CHKS)

        uids = [r[1] for r in results]
        div_weights = [r[2] for r in results]
        self.perfect_div_weight = dict()
        for uid, div_weight in zip(uids, div_weights):
            self.perfect_div_weight[uid] = div_weight
        results = [r[0] for r in results]

        fout = open(self.data_directory+UTIL +
                    f'parameter_{self.get_final_rec_name()}.pickle', "wb")
        pickle.dump(self.perfect_div_weight, fout)
        self.save_result(results, base=False)

    def geosoca(self):

        social_matrix = np.zeros((self.user_num, self.user_num))
        for uid1, friends in self.social_relations.items():
            for uid2 in friends:
                social_matrix[uid1, uid2] = 1.0
                social_matrix[uid2, uid1] = 1.0
        all_cats = set()
        for lid, cats in self.poi_cats.items():
            all_cats.update(cats)

        cat_to_int_id = dict()
        for i, cat in enumerate(all_cats):
            cat_to_int_id[cat] = i
        print(f"num of cats: {len(all_cats)}")
        poi_cat_matrix = np.zeros((self.poi_num, len(all_cats)))

        for lid, cats in self.poi_cats.items():
            for cat in cats:
                # little different from the original
                # in the original its not sum 1 but simple set to 1 always
                poi_cat_matrix[lid, cat_to_int_id[cat]] += 1.0
        self.cache[self.base_rec]['AKDE'] = AdaptiveKernelDensityEstimation(
            alpha=self.base_rec_parameters['alpha'])
        self.cache[self.base_rec]['SC'] = SocialCorrelation()
        self.cache[self.base_rec]['CC'] = CategoricalCorrelation()
        self.cache[self.base_rec]['AKDE'].precompute_kernel_parameters(
            self.training_matrix, self.poi_coos)
        self.cache[self.base_rec]['SC'].compute_beta(self.training_matrix, social_matrix)
        self.cache[self.base_rec]['CC'].compute_gamma(self.training_matrix, poi_cat_matrix)

        args = [(id(self),uid,) for uid in self.all_uids]

        with np.errstate(under='ignore'):
            results = run_parallel(self.run_geosoca, args, self.CHKS)
        self.save_result(results, base=True)

    @staticmethod
    def run_geosoca(recrunner_id, uid):
        self =ctypes.cast(recrunner_id,ctypes.py_object).value
        AKDE = self.cache[self.base_rec]['AKDE']
        SC = self.cache[self.base_rec]['SC']
        CC = self.cache[self.base_rec]['CC']

        if uid in self.ground_truth:

            overall_scores = normalize([AKDE.predict(uid, lid) * SC.predict(uid, lid) * CC.predict(uid, lid)
                                        if self.training_matrix[uid, lid] == 0 else -1
                                        for lid in self.all_lids])
            overall_scores = np.array(overall_scores)

            predicted = list(reversed(overall_scores.argsort()))[
                :self.base_rec_list_size]
            overall_scores = list(reversed(np.sort(overall_scores)))[
                :self.base_rec_list_size]
            return json.dumps({'user_id': uid, 'predicted': list(map(int, predicted)), 'score': list(map(float, overall_scores))})+"\n"
        self.not_in_ground_truth_message(uid)
        return ""

    @staticmethod
    def run_pdpgeocat(recrunner_id, uid):
        self =ctypes.cast(recrunner_id,ctypes.py_object).value
        actual = self.ground_truth[uid]
        max_predicted_val = -1
        max_predicted = None
        max_overall_scores = None
        if uid in self.ground_truth:
            x = 1
            r = 1/self.final_rec_parameters['interval']
            # for div_weight in np.append(np.arange(0, x, r),x):
            for div_weight in np.append(np.arange(0, x, r), x):
                # if not(div_weight==0 and div_weight!=div_geo_cat_weight):
                predicted = self.user_base_predicted_lid[uid][
                    0:self.base_rec_list_size]
                overall_scores = self.user_base_predicted_score[uid][
                    0:self.base_rec_list_size]

                predicted, overall_scores = gcobjfunc.geocat(uid, self.training_matrix, predicted, overall_scores,
                                                             self.poi_cats, self.poi_neighbors, self.final_rec_list_size, self.undirected_category_tree,
                                                             self.final_rec_parameters['div_geo_cat_weight'], div_weight,
                                                             'local_max')

                precision_val = metrics.precisionk(
                    actual, predicted[:self.final_rec_parameters['k']])
                if precision_val > max_predicted_val:
                    max_predicted_val = precision_val
                    max_predicted = predicted
                    max_overall_scores = overall_scores
                    max_div_weight = div_weight
                    # print("%d uid with geocatweight %f" % (uid,div_geo_cat_weight))
                    # print(self.perfect_parameter)
            predicted, overall_scores = max_predicted, max_overall_scores
            return json.dumps({'user_id': uid, 'predicted': list(map(int, predicted)), 'score': list(map(float, overall_scores))})+"\n", uid, max_div_weight
        self.not_in_ground_truth_message(uid)
        return ""


    @staticmethod
    def run_pm2(recrunner_id, uid):
        self =ctypes.cast(recrunner_id,ctypes.py_object).value
        if uid in self.ground_truth:
            predicted = self.user_base_predicted_lid[uid][
                0:self.base_rec_list_size]
            overall_scores = self.user_base_predicted_score[uid][
                0:self.base_rec_list_size]
            predicted, overall_scores = self.pm2.pm2(
                uid, predicted, overall_scores, self.final_rec_list_size)
            return json.dumps({'user_id': uid, 'predicted': list(map(int, predicted)), 'score': list(map(float, overall_scores))})+"\n"
        self.not_in_ground_truth_message(uid)
        return ""

    @staticmethod
    def run_usg(recrunner_id, uid, alpha, beta):
        self =ctypes.cast(recrunner_id,ctypes.py_object).value
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
        self.not_in_ground_truth_message(uid)
        return ""

    @staticmethod
    def run_mostpopular(recrunner_id, uid):
        self =ctypes.cast(recrunner_id,ctypes.py_object).value
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

            self.user_base_predicted_lid[uid] = predicted
            self.user_base_predicted_score[uid] = overall_scores
            return json.dumps({'user_id': uid, 'predicted': list(map(int, predicted)), 'score': list(map(float, overall_scores))})+"\n"
        self.not_in_ground_truth_message(uid)
        return ""

    @staticmethod
    def run_geocat(recrunner_id, uid):
        self =ctypes.cast(recrunner_id,ctypes.py_object).value
        if uid in self.ground_truth:
            predicted = self.user_base_predicted_lid[uid][
                0:self.base_rec_list_size]
            overall_scores = self.user_base_predicted_score[uid][
                0:self.base_rec_list_size]

            predicted, overall_scores = gcobjfunc.geocat(uid, self.training_matrix, predicted, overall_scores,
                                                         self.poi_cats, self.poi_neighbors, self.final_rec_list_size, self.undirected_category_tree,
                                                         self.final_rec_parameters[
                                                             'div_geo_cat_weight'], self.final_rec_parameters['div_weight'],
                                                         self.final_rec_parameters['heuristic'],
                                                         gcobjfunc.OBJECTIVE_FUNCTIONS.get(
                                                             self.final_rec_parameters['obj_func']),
                                                         self.final_rec_parameters['div_cat_weight'])

            return json.dumps({'user_id': uid, 'predicted': list(map(int, predicted)), 'score': list(map(float, overall_scores))})+"\n"
        self.not_in_ground_truth_message(uid)
        return ""

    @staticmethod
    def run_persongeocat(recrunner_id, uid):
        self =ctypes.cast(recrunner_id,ctypes.py_object).value
        if uid in self.ground_truth:
            predicted = self.user_base_predicted_lid[uid][
                0:self.base_rec_list_size]
            overall_scores = self.user_base_predicted_score[uid][
                0:self.base_rec_list_size]

            # start_time = time.time()
            div_geo_cat_weight = self.div_geo_cat_weight[uid]
            # if math.isnan(div_geo_cat_weight):
            #     print(f"User {uid} with nan value")
            div_weight = self.div_weight[uid]
            div_cat_weight = self.div_cat_weight[uid]
            predicted, overall_scores = gcobjfunc.geocat(uid, self.training_matrix, predicted, overall_scores,
                                                         self.poi_cats, self.poi_neighbors, self.final_rec_list_size, self.undirected_category_tree,
                                                         div_geo_cat_weight, div_weight,
                                                         'local_max',
                                                         gcobjfunc.OBJECTIVE_FUNCTIONS[self.final_rec_parameters['obj_func']],
                                                         div_cat_weight)

            # print("uid → %d, time → %fs" % (uid, time.time()-start_time))

            # predicted = np.array(predicted)[list(
            #     reversed(np.argsort(overall_scores)))]
            # overall_scores = list(reversed(np.sort(overall_scores)))

            return json.dumps({'user_id': uid, 'predicted': list(map(int, predicted)), 'score': list(map(float, overall_scores))})+"\n"
        self.not_in_ground_truth_message(uid)
        return ""

    @staticmethod
    def run_geodiv(recrunner_id, uid):
        self =ctypes.cast(recrunner_id,ctypes.py_object).value
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

            # predicted = np.array(predicted)[list(
            #     reversed(np.argsort(overall_scores)))]
            # overall_scores = list(reversed(np.sort(overall_scores)))

            return json.dumps({'user_id': uid, 'predicted': list(map(int, predicted)), 'score': list(map(float, overall_scores))})+"\n"
        self.not_in_ground_truth_message(uid)
        return ""

    @staticmethod
    def run_ld(recrunner_id, uid):
        self =ctypes.cast(recrunner_id,ctypes.py_object).value
        if uid in self.ground_truth:
            predicted = self.user_base_predicted_lid[uid][
                0:self.base_rec_list_size]
            overall_scores = self.user_base_predicted_score[uid][
                0:self.base_rec_list_size]

            predicted, overall_scores = gcobjfunc.ld(uid, self.training_matrix, predicted, overall_scores,
                                                     self.poi_cats, self.undirected_category_tree, self.final_rec_list_size,
                                                     self.final_rec_parameters['div_weight'])

            # predicted = np.array(predicted)[list(
            #     reversed(np.argsort(overall_scores)))]
            # overall_scores = list(reversed(np.sort(overall_scores)))

            return json.dumps({'user_id': uid, 'predicted': list(map(int, predicted)), 'score': list(map(float, overall_scores))})+"\n"
        self.not_in_ground_truth_message(uid)
        return ""

    @staticmethod
    def run_binomial(recrunner_id, uid):
        self =ctypes.cast(recrunner_id,ctypes.py_object).value
        if uid in self.ground_truth:
            predicted = self.user_base_predicted_lid[uid][
                0:self.base_rec_list_size]
            overall_scores = self.user_base_predicted_score[uid][
                0:self.base_rec_list_size]
            predicted, overall_scores = self.binomial.binomial(
                uid, predicted, overall_scores, self.final_rec_list_size)
            # predicted, overall_scores = gcobjfunc.ld(uid, self.training_matrix, predicted, overall_scores,
            #                                     self.poi_cats,self.undirected_category_tree,self.final_rec_list_size,
            #                                     self.final_rec_parameters['div_weight'])

            # predicted = np.array(predicted)[list(
            #     reversed(np.argsort(overall_scores)))]
            # overall_scores = list(reversed(np.sort(overall_scores)))

            return json.dumps({'user_id': uid, 'predicted': list(map(int, predicted)), 'score': list(map(float, overall_scores))})+"\n"
        self.not_in_ground_truth_message(uid)
        return ""

    def load_base_predicted(self):
        folds = [None]

        for self.fold in folds:

            result_file = open(
                self.data_directory+RECLIST+self.get_base_rec_file_name(), 'r')
            for i, line in enumerate(result_file):
                obj = json.loads(line)
                self.user_base_predicted_lid[obj['user_id']] = obj['predicted']
                self.user_base_predicted_score[obj['user_id']] = obj['score']
            self.recs_user_base_predicted_lid[self.get_base_rec_name(
            )] = self.user_base_predicted_lid
            self.recs_user_base_predicted_score[self.get_base_rec_name(
            )] = self.user_base_predicted_score

    def load_final_predicted(self):
        folds = [None]

        for self.fold in folds:
            result_file = open(
                self.data_directory+RECLIST+self.get_final_rec_file_name(), 'r')
            self.user_final_predicted_lid = dict()
            self.user_final_predicted_score = dict()
            for i, line in enumerate(result_file):
                obj = json.loads(line)
                self.user_final_predicted_lid[obj['user_id']
                                              ] = obj['predicted']
                self.user_final_predicted_score[obj['user_id']] = obj['score']
            self.recs_user_final_predicted_lid[self.get_final_rec_name(
            )] = self.user_final_predicted_lid
            self.recs_user_final_predicted_score[self.get_final_rec_name(
            )] = self.user_final_predicted_score

    def message_recommender(self, base):
        if base:
            print(f"{self.base_rec} base recommender")
        else:
            print(f"{self.final_rec} final recommender")
        self.print_parameters(base=base)
        print(f"Base rec list size = {self.base_rec_list_size}")
        print(f"Final rec list size = {self.final_rec_list_size}")

    def run_base_recommender(self, check_already_exists=False):

        if check_already_exists == True and os.path.exists(self.data_directory+RECLIST+self.get_base_rec_file_name()):
            print("recommender not going to be ran, already generated %s" %
                  (self.get_base_rec_name()))
            return
        base_recommender = self.BASE_RECOMMENDERS[self.base_rec]

        self.message_recommender(base=True)
        start_time = time.time()
        base_recommender()
        self.cache.clear()

        final_time = time.time()-start_time
        utils.create_path_to_file(
            self.data_directory+UTIL+f'run_time_{self.get_base_rec_name()}.txt')
        fout = open(self.data_directory+UTIL +
                    f'run_time_{self.get_base_rec_name()}.txt', "w")
        fout.write(str(final_time))
        fout.close()

    def run_final_recommender(self, check_already_exists=False):
        if check_already_exists == True and os.path.exists(self.data_directory+RECLIST+self.get_final_rec_file_name()):
            print("recommender not going to be ran, already generated %s" %
                  (self.get_final_rec_name()))
            return
        final_recommender = self.FINAL_RECOMMENDERS[self.final_rec]
        if len(self.user_base_predicted_lid) > 0:
            self.message_recommender(base=False)
            start_time = time.time()

            final_recommender()
            self.cache.clear()
            final_time = time.time()-start_time
            fout = open(self.data_directory+UTIL +
                        f'run_time_{self.get_final_rec_name()}.txt', "w")
            fout.write(str(final_time))
            fout.close()
        else:
            print("User base predicted list is empty")
        pass

    @staticmethod
    def eval(recrunner_id, uid, base, k):
        self =ctypes.cast(recrunner_id,ctypes.py_object).value
        if base:
            predictions = self.user_base_predicted_lid
            predicted = self.user_base_predicted_lid[uid]
        else:
            predictions = self.user_final_predicted_lid
            predicted = self.user_final_predicted_lid[uid]
        actual = self.ground_truth[uid]

        predicted_at_k = predicted[:k]
        precision_val = metrics.precisionk(actual, predicted_at_k)
        rec_val = metrics.recallk(actual, predicted_at_k)
        pr_val = metrics.prk(
            self.training_matrix[uid], predicted_at_k, self.poi_neighbors)
        ild_val = metrics.ildk(predicted_at_k, self.poi_cats,
                               self.undirected_category_tree)
        gc_val = metrics.gck(uid, self.training_matrix,
                             self.poi_cats, predicted_at_k)
        epc_val = self.epc_val
        f1_val = metrics.f1k(precision_val, rec_val)

        d = {'user_id': uid, 'precision': precision_val, 'recall': rec_val, 'pr': pr_val, 'ild': ild_val, 'gc': gc_val, 'epc': epc_val, 'f1': f1_val,
             # 'map':map_val,'ndcg':ndcg_val,
             # 'ildg': ildg_val
             }

        return json.dumps(d)+'\n'

    def get_file_name_metrics(self, base, k):
        if base:
            return self.data_directory+METRICS+self.get_base_rec_name()+f"_{str(k)}{R_FORMAT}"
        else:
            return self.data_directory+METRICS+self.get_final_rec_name()+f"_{str(k)}{R_FORMAT}"

    def eval_rec_metrics(self, *, base=False, METRICS_KS=experiment_constants.METRICS_K):

        folds = [None]
        for self.fold in folds:
            if base:
                predictions = self.recs_user_base_predicted_lid[self.get_base_rec_name(
                )]
                self.user_base_predicted_lid = predictions
            else:
                predictions = self.recs_user_final_predicted_lid[self.get_final_rec_name(
                )]
                self.user_final_predicted_lid = predictions
            all_uids = self.all_uids
            for i, k in enumerate(METRICS_KS):
                if (k <= self.final_rec_list_size and not base) or (k <= self.base_rec_list_size and base):
                    print(f"running metrics at @{k}")
                    self.epc_val = metrics.old_global_epck(
                        self.training_matrix, self.ground_truth, predictions, predictions.keys(), k)

                    utils.create_path_to_file(
                        self.get_file_name_metrics(base, k))
                    result_out = open(self.get_file_name_metrics(base, k), 'w')

                    self.message_recommender(base=base)

                    args = [(id(self),uid, base, k) for uid in all_uids]
                    results = run_parallel(self.eval, args, self.CHKSL)
                    print(pd.DataFrame([json.loads(result)
                          for result in results]).mean().T)
                    for json_string_result in results:
                        result_out.write(json_string_result)
                    result_out.close()
                else:
                    print(
                        f"Trying to evaluate list with @{k}, greather than final rec list size")

    def load_metrics(self, *, base=False, name_type=NameType.PRETTY, METRICS_KS=experiment_constants.METRICS_K, epc_group=False):
        if base:
            rec_using = self.base_rec
            if name_type == NameType.PRETTY:
                rec_short_name = self.get_base_rec_pretty_name()
            elif name_type == NameType.SHORT:
                rec_short_name = self.get_base_rec_short_name()
            elif name_type == NameType.FULL:
                rec_short_name = self.get_base_rec_name()
        else:
            rec_using = self.final_rec
            if name_type == NameType.BASE_PRETTY:
                rec_short_name = self.get_final_rec_pretty_name(
                )+'('+self.get_base_rec_pretty_name()+')'
            elif name_type == NameType.PRETTY:
                rec_short_name = self.get_final_rec_pretty_name()
            elif name_type == NameType.SHORT:
                rec_short_name = self.get_final_rec_short_name()
            elif name_type == NameType.CITY_BASE_PRETTY:
                rec_short_name = f"({CITIES_PRETTY[self.city]})"+self.get_final_rec_pretty_name(
                )+'('+self.get_base_rec_pretty_name()+')'
            elif name_type == NameType.FULL:
                rec_short_name = self.get_final_rec_name()

        print("Loading %s..." % (rec_short_name))

        if epc_group:
            fin = open(self.data_directory+UTIL +
                       f'groups_epc_{self.get_final_rec_name()}.pickle', "rb")
            self.groups_epc[rec_short_name] = pickle.load(fin)
            fin.close()
        self.metrics[rec_short_name] = {}
        self.metrics_cities[rec_short_name] = self.city
        for i, k in enumerate(METRICS_KS):
            try:
                result_file = open(self.get_file_name_metrics(base, k), 'r')

                self.metrics[rec_short_name][k] = []
                for i, line in enumerate(result_file):
                    obj = json.loads(line)
                    self.metrics[rec_short_name][k].append(obj)
            except Exception as e:
                print(e)
        if self.metrics[rec_short_name] == {}:
            del self.metrics[rec_short_name]
        return self.metrics[rec_short_name]

    def test_data(self):
        self.invalid_uids = []
        self.message_start_section("TESTING DATA SET")
        has_some_error_global = False
        for i in self.all_uids:
            has_some_error = False
            test_size = len(self.ground_truth.get(i, []))
            train_size = np.count_nonzero(self.training_matrix[i])
            if test_size == 0:
                print(f"user {i} with empty ground truth")
                has_some_error = True
                # remove from tests
                self.invalid_uids.append(i)
            if train_size == 0:
                print(
                    f"user {i} with empty training data!!!!! Really bad error")
                has_some_error = True
            if has_some_error:
                has_some_error_global = True
                print("Training size is %d, test size is %d" %
                      (train_size, test_size))
        for uid in self.invalid_uids:
            self.all_uids.remove(uid)

        if not has_some_error_global:
            print("No error encountered in base")

    def print_parameters(self, base):
        print_dict(self.get_base_parameters_descriptions()[self.base_rec])
        print_dict(self.base_rec_parameters)
        if base == False:
            print()
            print_dict(self.final_rec_parameters)

    def generate_general_user_data(self):
        preprocess_methods = [  # [None,'walk']
            # ,['poi_ild',None]
        ]
        for cat_div_method in [None]+CatDivPropensity.METHODS:
            for geo_div_method in [None]+GeoDivPropensity.METHODS:
                if cat_div_method != None or geo_div_method != None:
                    preprocess_methods.append([cat_div_method, geo_div_method])

        final_rec = self.final_rec
        self.final_rec = 'persongeocat'
        div_geo_cat_weight = dict()
        methods_columns = []
        for cat_div_method in [None]+CatDivPropensity.METHODS:
            for geo_div_method in [None]+GeoDivPropensity.METHODS:
                if cat_div_method != geo_div_method and ([cat_div_method, geo_div_method] in preprocess_methods):
                    self.final_rec_parameters = {
                        'cat_div_method': cat_div_method, 'geo_div_method': geo_div_method}
                    self.persongeocat_preprocess()
                    div_geo_cat_weight[(
                        cat_div_method, geo_div_method)] = self.div_geo_cat_weight
                    methods_columns.append(
                        "%s-%s" % (cat_div_method, geo_div_method))
        df = pd.DataFrame([],
                          columns=[
                              'visits', 'visits_mean',
                              'visits_std', 'cats_visited',
                              'cats_visited_mean', 'cats_visited_std',
                              'num_friends'
        ]
            + methods_columns
        )
        # self.persongeocat_preprocess()
        # div_geo_cat_weight = self.div_geo_cat_weight
        for uid in self.all_uids:
            # beta = self.perfect_parameter[uid]
            visited_lids = self.training_matrix[uid].nonzero()[0]
            visits_mean = self.training_matrix[uid, visited_lids].mean()
            visits_std = self.training_matrix[uid, visited_lids].std()
            visits = self.training_matrix[uid, visited_lids].sum()
            # uvisits = len(visited_lids)
            cats_visits = defaultdict(int)
            for lid in visited_lids:
                for cat in self.poi_cats[lid]:
                    cats_visits[cat] += 1
            cats_visits = np.array(list(cats_visits.values()))
            cats_visits_mean = cats_visits.mean()
            cats_visits_std = cats_visits.std()
            methods_values = []
            for cat_div_method in [None]+CatDivPropensity.METHODS:
                for geo_div_method in [None]+GeoDivPropensity.METHODS:
                    if cat_div_method != geo_div_method and ([cat_div_method, geo_div_method] in preprocess_methods):
                        methods_values.append(
                            div_geo_cat_weight[(cat_div_method, geo_div_method)][uid])
            num_friends = len(self.social_relations[uid])
            df.loc[uid] = [visits, visits_mean,
                           visits_std, len(cats_visits),
                           cats_visits_mean, cats_visits_std, num_friends] + methods_values
        df = pd.concat([df, self.user_data], axis=1)

        # poly = PolynomialFeatures(degree=1,interaction_only=False)
        # res_poly = poly.fit_transform(df)
        # df_poly = pd.DataFrame(res_poly, columns=poly.get_feature_names(df.columns))
        df_poly = df

        self.final_rec = final_rec
        return df_poly

    def gc(self):
        args = [(id(self),uid,) for uid in self.all_uids]
        results = run_parallel(self.run_gc, args, self.CHKS)
        self.save_result(results, base=False)

    @staticmethod
    def run_gc(recrunner_id, uid):
        self =ctypes.cast(recrunner_id,ctypes.py_object).value
        if uid in self.ground_truth:
            predicted = self.user_base_predicted_lid[uid][
                0:self.base_rec_list_size]
            overall_scores = self.user_base_predicted_score[uid][
                0:self.base_rec_list_size]

            predicted, overall_scores = gc_diversifier(uid, self.training_matrix,
                                                       predicted, overall_scores,
                                                       self.poi_cats,
                                                       self.undirected_category_tree,
                                                       self.final_rec_parameters['div_weight'],
                                                       self.final_rec_list_size)
            return json.dumps({'user_id': uid, 'predicted': list(map(int, predicted)), 'score': list(map(float, overall_scores))})+"\n"

        self.not_in_ground_truth_message(uid)
        return ""

    def random(self):
        args = [(id(self),uid,) for uid in self.all_uids]
        results = run_parallel(self.run_random, args, self.CHKS)
        self.save_result(results, base=False)

    @staticmethod
    def run_random(recrunner_id, uid):
        self =ctypes.cast(recrunner_id,ctypes.py_object).value
        if uid in self.ground_truth:
            predicted = self.user_base_predicted_lid[uid][
                0:self.base_rec_list_size]
            overall_scores = self.user_base_predicted_score[uid][
                0:self.base_rec_list_size]

            predicted, overall_scores = random_diversifier(predicted, overall_scores,
                                                           self.final_rec_list_size, self.final_rec_parameters['div_weight'])
            return json.dumps({'user_id': uid, 'predicted': list(map(int, predicted)), 'score': list(map(float, overall_scores))})+"\n"

        self.not_in_ground_truth_message(uid)
        return ""

    def print_base_info(self):
        val = list(map(len, self.poi_cats.values()))
        print("POI Categories")
        print(scipy.stats.describe(val))
        print('median', np.median(val))

        print("Visits")
        val = np.sum(self.training_matrix, axis=1)
        print(scipy.stats.describe(val))
        print('median', np.median(val))

        users_categories_visits = cat_utils.get_users_cat_visits(self.training_matrix,
                                                                 self.poi_cats)
        print('users_categories_visits')
        print(scipy.stats.describe(list(map(len, users_categories_visits))))
        print(np.median(list(map(len, users_categories_visits))))

    def geomf(self):
        geomf = GeoMF(**self.base_rec_parameters)
        geomf.train(self.training_matrix, self.poi_coos)
        self.cache['geomf'] = geomf
        args = [(id(self),uid,) for uid in self.all_uids]
        results = run_parallel(self.run_geomf, args, self.CHKS)
        self.save_result(results, base=True)

    @staticmethod
    def run_geomf(recrunner_id, uid):
        self =ctypes.cast(recrunner_id,ctypes.py_object).value
        geomf = self.cache['geomf']
        if uid in self.ground_truth:

            geomf_scores = geomf.predict(uid, self.all_lids)[0]
            # print(np.min(geomf_scores),np.max(geomf_scores),np.min(geomf.data['X'][uid]),np.max(geomf.data['X'][uid]))
            # min_score = np.min(geomf_scores)
            geomf_scores = geomf_scores-np.min(geomf_scores)
            geomf_scores = geomf_scores/np.max(geomf_scores)
            # print(geomf_scores.min(),geomf_scores.max())
            overall_scores = normalize([geomf_scores[lid]
                                        if self.training_matrix[uid, lid] == 0 else -1
                                        for lid in self.all_lids])
            overall_scores = np.array(overall_scores)
            # overall_scores = overall_scores-np.min(overall_scores)
            # overall_scores = overall_scores/np.max(overall_scores)

            predicted = list(reversed(overall_scores.argsort()))[
                :self.base_rec_list_size]
            overall_scores = list(reversed(np.sort(overall_scores)))[
                :self.base_rec_list_size]
            return json.dumps({'user_id': uid, 'predicted': list(map(int, predicted)), 'score': list(map(float, overall_scores))})+"\n"
        self.not_in_ground_truth_message(uid)
        return ""

    def geodiv2020(self):
        geodiv2020 = GeoDiv2020(**self.final_rec_parameters)
        geodiv2020.train(self.training_matrix, self.poi_coos)
        self.cache['geodiv2020'] = geodiv2020
        args = [(id(self),uid,) for uid in self.all_uids]
        results = run_parallel(self.run_geodiv2020, args, self.CHKS)
        self.save_result(results, base=False)

    @staticmethod
    def run_geodiv2020(recrunner_id, uid):
        self =ctypes.cast(recrunner_id,ctypes.py_object).value
        geodiv2020 = self.cache['geodiv2020']
        if uid in self.ground_truth:
            predicted = self.user_base_predicted_lid[uid][
                0:self.base_rec_list_size]
            overall_scores = self.user_base_predicted_score[uid][
                0:self.base_rec_list_size]

            predicted, overall_scores = geodiv2020.predict(
                uid, predicted, overall_scores, self.final_rec_list_size)
            assert(self.final_rec_list_size == len(predicted))

            return json.dumps({'user_id': uid, 'predicted': list(map(int, predicted)), 'score': list(map(float, overall_scores))})+"\n"
        self.not_in_ground_truth_message(uid)
        return ""
