# import os
# import sys
# module_path = os.path.abspath(os.path.join('.'))
# if module_path not in sys.path:
#     sys.path.append(module_path)
#print(sys.path)

LATEX_HEADER = r"""\documentclass{article}

\title{Geocat}
\author{heitorwerneck }
\date{January 2020}

\usepackage{natbib}
\usepackage{graphicx}
\usepackage[utf8]{inputenc}
\usepackage{xcolor}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{underscore}
\usepackage[margin=0.5in]{geometry}
\begin{document}

\maketitle

\section{Introduction}
"""

LATEX_FOOT = r"""\bibliographystyle{plain}
\bibliography{references}
\end{document}"""

from abc import ABC, abstractmethod
from collections import defaultdict, OrderedDict
import pickle
from concurrent.futures import ProcessPoolExecutor
import json
import time
from datetime import datetime
import itertools
import multiprocessing
from collections import Counter
import os

import inquirer
import numpy as np
from tqdm import tqdm
from scipy.stats import describe
import matplotlib.pyplot as plt
from cycler import cycler
# PALETTE = plt.get_cmap('Greys')
# monochrome = (cycler('color', ['k']) * cycler('linestyle', ['-.', '--','-', ':', ]) * cycler('marker', [',','.','^']))
plt.rcParams['font.size'] = 11
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.framealpha'] = 0.5
# plt.rcParams['axes.grid'] = True
# plt.rcParams['axes.prop_cycle'] = monochrome
# plt.rcParams['axes.spines.top'] = False
# plt.rcParams['axes.spines.right'] = False

BAR_EDGE_COLOR = 'black'
linestyle_tuple = {
     'loosely dotted':        (0, (1, 10)),
     'dotted':                (0, (1, 1)),
     'densely dotted':        (0, (1, 1)),
     'loosely dashed':        (0, (5, 10)),
     'dashed':                (0, (5, 5)),
     'densely dashed':        (0, (5, 1)),
     'loosely dashdotted':    (0, (3, 10, 1, 10)),
     'dashdotted':            (0, (3, 5, 1, 5)),
     'densely dashdotted':    (0, (3, 1, 1, 1)),
     'dashdotdotted':         (0, (3, 5, 1, 5, 1, 5)),
     'loosely dashdotdotted': (0, (3, 10, 1, 10, 1, 10)),
     'densely dashdotdotted': (0, (3, 1, 1, 1, 1, 1))}
def gen_line_cycle(num=6):
    start = 0
    final = 0.6
    arange = np.linspace(start,final,num)

    color_cycler = cycler('color', reversed(list(map(str,arange))))
    linestyle = cycler('linestyle', ['-','--','-.',
                                     linestyle_tuple['dashdotted'],
                                     linestyle_tuple['densely dashdotdotted'],
                                     ':',
                                     linestyle_tuple['dashdotdotted'],
                                     linestyle_tuple['loosely dashdotted'],
    ][:num])
    linewidth = cycler('linewidth',[3])
    markerwidth = cycler('markersize',[10])
    
    print(list(color_cycler))
    marker = cycler('marker', ['^','v','s','x','d','o','1','h','P','*'][:num])
    return (color_cycler + linestyle + marker) * linewidth * markerwidth

def gen_bar_cycle(num=6):
    
    hatch_cycler = cycler('hatch', ['///', '--', '...','\///', 'xxx', '\\\\','None','*'][:num])
    base_cycler = cycler('zorder', [10])*cycler('edgecolor',[BAR_EDGE_COLOR])
    color_cycler = cycler('color', reversed([str(i/(num-1)) for i in range(num)]))
    bar_cycle = (hatch_cycler + color_cycler)*base_cycler
    return bar_cycle
GEOCAT_BAR_STYLE = {'color': 'k', 'zorder': 10,'edgecolor':BAR_EDGE_COLOR}


# plt.rcParams['axes.spines.bottom'] = False
# plt.rcParams['axes.spines.left'] = False

# plt.rcParams['axes.grid'] = True
MPL_ALPHA = 0.5
# plt.rcParams['axes.grid.alpha'] = 0.5
from pympler import asizeof
import scipy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing
from sklearn import svm
from sklearn.preprocessing import PolynomialFeatures
from sklearn import neighbors
import imblearn
import imblearn.datasets
import imblearn.over_sampling
import imblearn.under_sampling
import scipy.stats

import cat_utils
from usg.UserBasedCF import UserBasedCF
from usg.FriendBasedCF import FriendBasedCF
from usg.PowerLaw import PowerLaw
import geocat.objfunc as gcobjfunc
from pgc.GeoDivPropensity import GeoDivPropensity
from pgc.CatDivPropensity import CatDivPropensity
from constants import experiment_constants, METRICS_PRETTY, RECS_PRETTY, CITIES_PRETTY
import metrics
from geocat.Binomial import Binomial
from geocat.Pm2 import Pm2
from parallel_util import run_parallel
from geosoca.AdaptiveKernelDensityEstimation import AdaptiveKernelDensityEstimation
from geosoca.SocialCorrelation import SocialCorrelation
from geosoca.CategoricalCorrelation import CategoricalCorrelation

CMAP_NAME = 'Set1'

DATA_DIRECTORY = '../data'  # directory with all data

R_FORMAT = '.json'  # Results format is json, metrics, rec lists, etc
D_FORMAT = '.pickle'  # data set format is pickle

TRAIN = 'checkin/train/'  # train data sets
TEST = 'checkin/test/'  # test data sets
POI = 'poi/'  # poi data sets with cats and coos
POI_FULL = 'poi_full/'  # poi data sets with cats full without preprocessing
USER_FRIEND = 'user/friend/'  # users and friends
USER = 'user/' # user general data
NEIGHBOR = 'neighbor/'  # neighbors of pois

METRICS = 'result/metrics/'
RECLIST = 'result/reclist/'
IMG = 'result/img/'
UTIL = 'result/util/'

#CHKS = 40 # chunk size for process pool executor
#CHKSL = 200 # chunk size for process pool executor largest

regressor_predictors = {
    # 'Linear regressor' : LinearRegression(),
    # 'Neural network regressor' : MLPRegressor(hidden_layer_sizes=(20,20,20,20)),
}

classifier_predictors = {
    'RFC' : RandomForestClassifier(n_estimators=500),
    # 'MLPC' : MLPClassifier(hidden_layer_sizes=(20,20,20,20,20,20)),
    'SVM' : svm.SVC(),
    'KNN' : neighbors.KNeighborsClassifier(),
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
    _instance = None
    def save_result(self,results,base=True):
        if base:
            result_out = open(self.data_directory+RECLIST+ self.get_base_rec_file_name(), 'w')
        else:
            result_out = open(self.data_directory+RECLIST+ self.get_final_rec_file_name(), 'w')
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
            "usg": self.usg,
            "geosoca": self.geosoca,
        }
        self.FINAL_RECOMMENDERS = {
            "geocat": self.geocat,
            "pgeocat": self.persongeocat,
            "geodiv": self.geodiv,
            "ld": self.ld,
            "binomial": self.binomial,
            "pm2": self.pm2,
            "perfectpgeocat": self.perfectpersongeocat,
            "pdpgeocat": self.pdpgeocat,
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
        self.metrics_name = ['precision', 'recall', 'gc', 'ild', 'epc','pr']
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
                # default value
                source_of_parameters_value = final_parameters
            else:
                # input value
                source_of_parameters_value = parameters
            parameters_result[parameter] = source_of_parameters_value[parameter]
        # filtering special cases
        if self.final_rec == 'geocat' and parameters_result['obj_func'] != 'cat_weight':
            parameters_result['div_cat_weight'] = None
        if self.final_rec == 'geocat' and parameters_result['obj_func'] == 'cat_weight' and parameters_result['div_cat_weight'] == None:
            print("Auto setting div_cat_weight to 0.95")
            parameters_result['div_cat_weight'] = 0.95
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
            "usg": {'alpha': 0.1, 'beta': 0.1, 'eta': 0.05},
            "geosoca": {'alpha': 0.5},
        }

    @staticmethod
    def get_final_parameters():
        return  {
            "geocat": {'div_weight':0.75,'div_geo_cat_weight':0.25, 'heuristic': 'local_max', 'obj_func': 'cat_weight', 'div_cat_weight': 0.05},
            "pgeocat": {'div_weight':0.75,'cat_div_method': None, 'geo_div_method': 'walk', 'obj_func': 'cat_weight', 'div_cat_weight':0.05, 'bins': 3,
                        'norm_method': 'median_quad'},
            "geodiv": {'div_weight':0.5},
            "ld": {'div_weight':0.25},
            "binomial": {'alpha': 0.5, 'div_weight': 0.75},
            "pm2": {'lambda': 1},
            "perfectpgeocat": {'k': 10,'interval': 2,'div_weight': 0.75},
            "pdpgeocat": {'k': 10,'interval': 2,'div_geo_cat_weight': 0.75},
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
            return f"{self.get_base_rec_short_name()}_{self.final_rec}_{self.final_rec_parameters['heuristic']}_{self.final_rec_parameters['obj_func']}"
        elif self.final_rec == 'persongeocat':
            string = f"{self.get_base_rec_short_name()}_{self.final_rec}"
            # if self.final_rec_parameters['cat_div_method']:
            string += f"_{self.final_rec_parameters['cat_div_method']}"
            # if self.final_rec_parameters['geo_div_method']:
            string += f"_{self.final_rec_parameters['geo_div_method']}"
            return string
        elif self.final_rec == 'geodiv':
            return f"{self.get_base_rec_short_name()}_{self.final_rec}"
        else:
            return f"{self.get_base_rec_short_name()}_{self.final_rec}"

    def get_base_rec_pretty_name(self):
        return RECS_PRETTY[self.base_rec]

    def get_final_rec_pretty_name(self):
        return RECS_PRETTY[self.final_rec]

    def get_base_rec_file_name(self):
        return self.get_base_rec_name()+".json"

    def get_final_rec_file_name(self):
        return self.get_final_rec_name()+".json"

    def get_base_metrics_name(self):
        return self.data_directory+METRICS+self.get_base_rec_name()+f"{R_FORMAT}"

    def get_final_metrics_name(self):
        return self.data_directory+METRICS+self.get_final_rec_name()+f"{R_FORMAT}"

    def load_base(self,user_data=True):
        CITY = self.city

        print(f"{CITY} city base loading")

        # Train load
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
#4.3 parametros corretos
        # self.poi_cats_full = {}
        # for poi_id, poi in pickle.load(open(self.data_directory+POI_FULL+CITY+".pickle", "rb")).items():
        #     self.poi_cats_full[poi_id] = poi['categories']

        # Social relations load
        self.social_relations = defaultdict(list)
        for user_id, friends in pickle.load(open(self.data_directory+USER_FRIEND+CITY+".pickle", "rb")).items():
            self.social_relations[user_id] = friends
        
        self.social_relations = dict(self.social_relations)
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

        # print(len(self.data_checkin_train))

        # poi neighbors load
        self.poi_neighbors = pickle.load(
            open(self.data_directory+NEIGHBOR+CITY+".pickle", "rb"))
        print(f"{CITY} city base loaded")
        self.all_uids = list(range(user_num))
        self.all_lids = list(range(poi_num))

        if user_data:
            self.user_data = pickle.load(open(self.data_directory+USER+CITY+'.pickle','rb'))

        self.test_data()

        print("Removing invalid users")

        self.training_matrix = self.training_matrix[self.all_uids]

        uid_to_int = dict()
        for i, uid in enumerate(self.all_uids):
            uid_to_int[uid] = i

        for uid in self.invalid_uids:
            del self.social_relations[uid]
            del self.user_data[uid]

        self.ground_truth = dict((uid_to_int[key], value) for (key, value) in self.ground_truth.items())
        self.social_relations = dict((uid_to_int[key], value) for (key, value) in self.social_relations.items())
        self.user_data = dict((uid_to_int[key], value) for (key, value) in self.user_data.items())

        
        for uid, i in uid_to_int.items():
            # self.ground_truth[uid_to_int[uid]] = self.ground_truth.pop(uid)
            # self.social_relations[uid_to_int[uid]] = self.social_relations.pop(uid)
            self.social_relations[i] = [uid_to_int[friend_uid] for friend_uid in self.social_relations[i]
                                          if friend_uid in uid_to_int]
            # self.user_data[uid_to_int[uid]] = self.user_data.pop(uid)


        self.all_uids = [uid_to_int[uid] for uid in self.all_uids]
        self.user_num = len(self.all_uids)
        print("Finish removing invalid users")

        if user_data:
            self.user_data = pd.DataFrame(self.user_data).T
            self.user_data['yelping_since'] = self.user_data['yelping_since'].apply(lambda date: pd.Timestamp(date).year)
            years = list(range(2004,2019))
            self.user_data = pd.get_dummies(self.user_data, columns=['yelping_since'])
            for year in years:
                if f'yelping_since_{year}' not in self.user_data.columns:
                    self.user_data[f'yelping_since_{year}'] = 0
            print("User data memory usage:",asizeof.asizeof(self.user_data)/1024**2,"MB")

        self.CHKS = int(len(self.all_uids)/multiprocessing.cpu_count()/8)
        self.CHKSL = int(len(self.all_uids)/multiprocessing.cpu_count())
        self.welcome_load()
        


    def welcome_load(self):
        self.message_start_section("LOAD FINAL MESSAGE")
        print('user num: %d, poi num: %d, checkin num: %d' % (self.training_matrix.shape[0],self.training_matrix.shape[1],self.training_matrix.sum().sum()))
        print("Chunk size set to %d for this base" %(self.CHKS))
        print("Large chunk size set to %d for this base" %(self.CHKSL))

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
        alpha = self.base_rec_parameters['alpha']
        beta = self.base_rec_parameters['beta']

        U = UserBasedCF()
        S = FriendBasedCF(eta=self.base_rec_parameters['eta'])
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
        # results = []
        # for i, arg in enumerate(args):
        #     print("%d/%d" % (i,len(args)))
        #     results.append(self.run_usg(*arg))
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

    def persongeocat_preprocess(self):
        if self.final_rec_parameters['cat_div_method'] in list(classifier_predictors.keys()) and self.final_rec_parameters['geo_div_method'] == None:
            bckp = vars(self).copy()

            df_test = self.generate_general_user_data()

            self.city = 'madison'
            self.load_base()
            self.load_perfect()
            perfect_parameter_train = self.perfect_parameter
            df = self.generate_general_user_data()

            for key, val in bckp.items():
                vars(self)[key] = val

            # print(len(self.perfect_parameter))
            # print(len(self.all_uids))

            X_train, X_test, y_train = df.to_numpy(),df_test.to_numpy(), perfect_parameter_train

            lab_enc = preprocessing.LabelEncoder()

            encoded_y_train = lab_enc.fit_transform(y_train)

            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

            name = self.final_rec_parameters['cat_div_method']
            cp = classifier_predictors[name]
            cp.fit(X_train,encoded_y_train)
            pred_cp = cp.predict(X_test)
            print("-------",name,"-------")
            self.div_geo_cat_weight = lab_enc.inverse_transform(pred_cp)
            self.div_weight=np.ones(len(self.div_geo_cat_weight))*self.final_rec_parameters['div_weight']
            return


        if self.final_rec_parameters['geo_div_method'] != None:
            print("Computing geographic diversification propensity")
            self.pgeo_div_runner = GeoDivPropensity.getInstance(self.training_matrix, self.poi_coos,
                                                                self.poi_cats,self.undirected_category_tree,
                                                                self.final_rec_parameters['geo_div_method'])
            self.geo_div_propensity = self.pgeo_div_runner.compute_div_propensity()

        if self.final_rec_parameters['cat_div_method'] != None:
            self.pcat_div_runner = CatDivPropensity.getInstance(
                self.training_matrix,
                self.undirected_category_tree,
                self.final_rec_parameters['cat_div_method'],
                self.poi_cats)
            print("Computing categoric diversification propensity with",
                self.final_rec_parameters['cat_div_method'])
            self.cat_div_propensity=self.pcat_div_runner.compute_div_propensity()

        if self.final_rec_parameters['norm_method'] == 'default':
            if self.final_rec_parameters['cat_div_method'] == None:
                self.div_geo_cat_weight=self.geo_div_propensity
                self.div_weight=np.ones(len(self.div_geo_cat_weight))*self.final_rec_parameters['div_weight']
            elif self.final_rec_parameters['geo_div_method'] == None:
                self.div_geo_cat_weight=1-self.cat_div_propensity
                self.div_weight=np.ones(len(self.div_geo_cat_weight))*self.final_rec_parameters['div_weight']
            else:
                self.div_geo_cat_weight=(self.geo_div_propensity)/(self.geo_div_propensity+self.cat_div_propensity)
                self.div_weight=np.ones(len(self.div_geo_cat_weight))*self.final_rec_parameters['div_weight']
                self.div_weight[(self.geo_div_propensity==0) & (self.cat_div_propensity==0)] = 0
        elif self.final_rec_parameters['norm_method'] == 'median_quad':
            cat_div_propensity = self.cat_div_propensity
            geo_div_propensity = self.geo_div_propensity
            groups = dict()
            groups['geo_preference'] = (cat_div_propensity <= cat_median) & (geo_div_propensity > geo_median)
            groups['no_preference'] = ((cat_div_propensity <= cat_median) & (geo_div_propensity <= geo_median)) | ((cat_div_propensity >= cat_median) & (geo_div_propensity >= geo_median))
            groups['cat_preference'] = (cat_div_propensity > cat_median) & (geo_div_propensity <= geo_median)
            self.div_geo_cat_weight=np.zeros(self.training_matrix.shape[0])
            self.div_geo_cat_weight[groups['geo_preference']] = 0
            self.div_geo_cat_weight[groups['no_preference']] = 0.5
            self.div_geo_cat_weight[groups['cat_preference']] = 1
            self.div_weight=np.ones(len(self.div_geo_cat_weight))*self.final_rec_parameters['div_weight']
        if self.final_rec_parameters['bins'] != None:
            bins = np.append(np.arange(0,1,1/(self.final_rec_parameters['bins']-1)),1)
            centers = (bins[1:]+bins[:-1])/2
            self.div_geo_cat_weight = bins[np.digitize(self.div_geo_cat_weight, centers)]
        # fout = open(self.data_directory+UTIL+f'parameter_{self.get_final_rec_name()}.pickle',"wb")
        # pickle.dump(self.div_geo_cat_weight,fout)
        # fout.close()

    def persongeocat(self):
        self.persongeocat_preprocess()
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
        del self.binomial

    def pm2(self):
        self.pm2 = Pm2(self.training_matrix,self.poi_cats,self.final_rec_parameters['lambda'])
        # for uid in self.all_uids:
        #     self.run_pm2(uid)
        args=[(uid,) for uid in self.all_uids]
        results = run_parallel(self.run_pm2,args,self.CHKS)
        self.save_result(results,base=False)
        del self.pm2

    def perfectpersongeocat(self):
        args=[(uid,) for uid in self.all_uids]
        results = run_parallel(self.run_perfectpersongeocat,args,self.CHKS)

        uids = [r[1] for r in results]
        div_geo_cat_weights = [r[2] for r in results]
        self.perfect_parameter = dict()
        for uid,div_geo_cat_weight in zip(uids,div_geo_cat_weights):
            self.perfect_parameter[uid] = div_geo_cat_weight
        # print(self.perfect_parameter)
        results = [r[0] for r in results]

        fout = open(self.data_directory+UTIL+f'parameter_{self.get_final_rec_name()}.pickle',"wb")
        pickle.dump(self.perfect_parameter,fout)
        self.save_result(results,base=False)

    def pdpgeocat(self):
        args=[(uid,) for uid in self.all_uids]
        results = run_parallel(self.run_pdpgeocat,args,self.CHKS)

        uids = [r[1] for r in results]
        div_weights = [r[2] for r in results]
        self.perfect_div_weight = dict()
        for uid,div_weight in zip(uids,div_weights):
            self.perfect_div_weight[uid] = div_weight
        # print(self.perfect_parameter)
        results = [r[0] for r in results]

        fout = open(self.data_directory+UTIL+f'parameter_{self.get_final_rec_name()}.pickle',"wb")
        pickle.dump(self.perfect_div_weight,fout)
        self.save_result(results,base=False)

    def pdpgeocat(self):
        args=[(uid,) for uid in self.all_uids]
        results = run_parallel(self.run_pdpgeocat,args,self.CHKS)

        uids = [r[1] for r in results]
        div_weights = [r[2] for r in results]
        self.perfect_div_weight = dict()
        for uid,div_weight in zip(uids,div_weights):
            self.perfect_div_weight[uid] = div_weight
        # print(self.perfect_parameter)
        results = [r[0] for r in results]

        fout = open(self.data_directory+UTIL+f'parameter_{self.get_final_rec_name()}.pickle',"wb")
        pickle.dump(self.perfect_div_weight,fout)
        self.save_result(results,base=False)

    def geosoca(self):

        #training_matrix = self.training_matrix.astype(np.int64)
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
        poi_cat_matrix = np.zeros((self.poi_num,len(all_cats)))

        for lid, cats in self.poi_cats.items():
            for cat in cats:
                # little different from the original
                # in the original its not sum 1 but simple set to 1 always
                poi_cat_matrix[lid, cat_to_int_id[cat]] += 1.0
        self.AKDE = AdaptiveKernelDensityEstimation(alpha=self.base_rec_parameters['alpha'])
        self.SC = SocialCorrelation()
        self.CC = CategoricalCorrelation()
        self.AKDE.precompute_kernel_parameters(self.training_matrix, self.poi_coos)
        self.SC.compute_beta(self.training_matrix, social_matrix)
        self.CC.compute_gamma(self.training_matrix, poi_cat_matrix)

        args=[(uid,) for uid in self.all_uids]
        results = run_parallel(self.run_geosoca,args,self.CHKS)
        self.save_result(results,base=True)

    @classmethod
    def run_geosoca(cls, uid):
        self = cls.getInstance()
        AKDE = self.AKDE
        SC = self.SC
        CC = self.CC

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
        self.not_in_ground_truth_message()
        return ""

    @classmethod
    def run_pdpgeocat(cls, uid):
        self = cls.getInstance()
        actual=self.ground_truth[uid]
        max_predicted_val = -1
        max_predicted = None
        max_overall_scores = None
        if uid in self.ground_truth:
            x = 1
            r = 1/self.final_rec_parameters['interval']
            # for div_weight in np.append(np.arange(0, x, r),x):
            for div_weight in np.append(np.arange(0, x, r),x):
                # if not(div_weight==0 and div_weight!=div_geo_cat_weight):
                predicted = self.user_base_predicted_lid[uid][
                    0:self.base_rec_list_size]
                overall_scores = self.user_base_predicted_score[uid][
                    0:self.base_rec_list_size]

                predicted, overall_scores = gcobjfunc.geocat(uid, self.training_matrix, predicted, overall_scores,
                                                            self.poi_cats, self.poi_neighbors, self.final_rec_list_size, self.undirected_category_tree,
                                                            self.final_rec_parameters['div_geo_cat_weight'],div_weight,
                                                                'local_max')

                precision_val=metrics.precisionk(actual, predicted[:self.final_rec_parameters['k']])
                if precision_val > max_predicted_val:
                    max_predicted_val = precision_val
                    max_predicted = predicted
                    max_overall_scores = overall_scores
                    max_div_weight = div_weight
                    # print("%d uid with geocatweight %f" % (uid,div_geo_cat_weight))
                    # print(self.perfect_parameter)
            predicted, overall_scores = max_predicted, max_overall_scores
            return json.dumps({'user_id': uid, 'predicted': list(map(int, predicted)), 'score': list(map(float, overall_scores))})+"\n", uid, max_div_weight
        self.not_in_ground_truth_message()
        return ""

    @classmethod
    def run_perfectpersongeocat(cls, uid):
        self = cls.getInstance()
        actual=self.ground_truth[uid]
        max_predicted_val = -1
        max_predicted = None
        max_overall_scores = None
        if uid in self.ground_truth:
            x = 1
            r = 1/self.final_rec_parameters['interval']
            # for div_weight in np.append(np.arange(0, x, r),x):
            for div_geo_cat_weight in np.append(np.arange(0, x, r),x):
                # if not(div_weight==0 and div_weight!=div_geo_cat_weight):
                predicted = self.user_base_predicted_lid[uid][
                    0:self.base_rec_list_size]
                overall_scores = self.user_base_predicted_score[uid][
                    0:self.base_rec_list_size]

                predicted, overall_scores = gcobjfunc.geocat(uid, self.training_matrix, predicted, overall_scores,
                                                            self.poi_cats, self.poi_neighbors, self.final_rec_list_size, self.undirected_category_tree,
                                                            div_geo_cat_weight,self.final_rec_parameters['div_weight'],
                                                                'local_max')

                precision_val=metrics.precisionk(actual, predicted[:self.final_rec_parameters['k']])
                if precision_val > max_predicted_val:
                    max_predicted_val = precision_val
                    max_predicted = predicted
                    max_overall_scores = overall_scores
                    max_div_geo_cat_weight = div_geo_cat_weight
                    # print("%d uid with geocatweight %f" % (uid,div_geo_cat_weight))
                    # print(self.perfect_parameter)
            predicted, overall_scores = max_predicted, max_overall_scores
            return json.dumps({'user_id': uid, 'predicted': list(map(int, predicted)), 'score': list(map(float, overall_scores))})+"\n", uid, max_div_geo_cat_weight
        self.not_in_ground_truth_message()
        return ""

    @classmethod
    def run_pm2(cls,uid):
        self = cls.getInstance()
        if uid in self.ground_truth:
            predicted = self.user_base_predicted_lid[uid][
                0:self.base_rec_list_size]
            overall_scores = self.user_base_predicted_score[uid][
                0:self.base_rec_list_size]
            predicted, overall_scores=self.pm2.pm2(uid,predicted,overall_scores,self.final_rec_list_size)
            return json.dumps({'user_id': uid, 'predicted': list(map(int, predicted)), 'score': list(map(float, overall_scores))})+"\n"
        self.not_in_ground_truth_message()
        return ""
    
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

            predicted, overall_scores = gcobjfunc.geocat(uid, self.training_matrix, predicted, overall_scores,
                                                         self.poi_cats, self.poi_neighbors, self.final_rec_list_size, self.undirected_category_tree,
                                                         self.final_rec_parameters['div_geo_cat_weight'],self.final_rec_parameters['div_weight'],
                                                         self.final_rec_parameters['heuristic'],
                                                         gcobjfunc.OBJECTIVE_FUNCTIONS[self.final_rec_parameters['obj_func']],
                                                         self.final_rec_parameters['div_cat_weight'])

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
            div_geo_cat_weight=self.div_geo_cat_weight[uid]
            div_weight=self.div_weight[uid]
            predicted, overall_scores = gcobjfunc.geocat(uid, self.training_matrix, predicted, overall_scores,
                                                         self.poi_cats, self.poi_neighbors, self.final_rec_list_size, self.undirected_category_tree,
                                                         div_geo_cat_weight,div_weight,
                                                         'local_max',
                                                         gcobjfunc.OBJECTIVE_FUNCTIONS[self.final_rec_parameters['obj_func']],
                                                         self.final_rec_parameters['div_cat_weight'])

            # print("uid → %d, time → %fs" % (uid, time.time()-start_time))

            # predicted = np.array(predicted)[list(
            #     reversed(np.argsort(overall_scores)))]
            # overall_scores = list(reversed(np.sort(overall_scores)))

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

            # predicted = np.array(predicted)[list(
            #     reversed(np.argsort(overall_scores)))]
            # overall_scores = list(reversed(np.sort(overall_scores)))

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

            # predicted = np.array(predicted)[list(
            #     reversed(np.argsort(overall_scores)))]
            # overall_scores = list(reversed(np.sort(overall_scores)))

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

            # predicted = np.array(predicted)[list(
            #     reversed(np.argsort(overall_scores)))]
            # overall_scores = list(reversed(np.sort(overall_scores)))

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
        print(f"Base rec list size = {self.base_rec_list_size}")
        print(f"Final rec list size = {self.final_rec_list_size}")

    def run_base_recommender(self):
        base_recommender=self.BASE_RECOMMENDERS[self.base_rec]
        self.message_recommender(base=True)
        base_recommender()

    def run_final_recommender(self,check_already_exists=False):
        if check_already_exists == True and os.path.exists(self.data_directory+RECLIST+self.get_final_rec_file_name()):
            print("recommender not going to be ran, already generated %s" % (self.get_final_rec_name()))
            return
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
    @classmethod
    def eval(cls,uid,base,k):
        self = cls.getInstance()
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

        map_val = metrics.mapk(actual,predicted_at_k,k)
        ndcg_val = metrics.ndcgk(actual,predicted_at_k,k)
        ildg_val = metrics.ildgk(predicted_at_k,self.poi_coos)
        # else:
        #     epc_val=self.epc_val
        # if uid == max(self.all_uids):
        #     del self.epc_val
        # this epc is maded like vargas, recsys'11
        #epc_val=metrics.epck(predicted_at_k,actual,uid,self.training_matrix)
        
        d={'user_id':uid,'precision':precision_val,'recall':rec_val,'pr':pr_val,'ild':ild_val,'gc':gc_val,'epc':epc_val,
           'map':map_val,'ndcg':ndcg_val,
           'ildg': ildg_val}

        return json.dumps(d)+'\n'

    def get_file_name_metrics(self,base,k):
        if base:
            return self.data_directory+"result/metrics/"+self.get_base_rec_name()+f"_{str(k)}{R_FORMAT}"
        else:
            return self.data_directory+"result/metrics/"+self.get_final_rec_name()+f"_{str(k)}{R_FORMAT}"

    def eval_rec_metrics(self,*,base=False):
        if base:
            predictions = self.user_base_predicted_lid
        else:
            predictions = self.user_final_predicted_lid

        for i,k in enumerate(experiment_constants.METRICS_K):
            print(f"running metrics at @{k}")
            self.epc_val = metrics.old_global_epck(self.training_matrix,self.ground_truth,predictions,self.all_uids,k)

            # if base:
            result_out = open(self.get_file_name_metrics(base,k), 'w')
            # else:
            #     result_out = open(self.data_directory+"result/metrics/"+self.get_final_rec_name()+f"_{str(k)}{R_FORMAT}", 'w')
            
            self.message_recommender(base=base)

            args=[(uid,base,k) for uid in self.all_uids]
            results = run_parallel(self.eval,args,self.CHKSL)
            
            for json_string_result in results:
                result_out.write(json_string_result)
            result_out.close()

    def load_metrics(self,*,base=False,pretty_name=False,short_name=True):
        if base:
            rec_using=self.base_rec
            if pretty_name:
                rec_short_name=self.get_base_rec_pretty_name()
            elif short_name:
                rec_short_name=self.get_base_rec_short_name()
            else:
                rec_short_name=self.get_base_rec_name()
        else:
            rec_using=self.final_rec
            if pretty_name:
                rec_short_name=self.get_final_rec_pretty_name()
            elif short_name:
                rec_short_name=self.get_final_rec_short_name()
            else:
                rec_short_name=self.get_final_rec_name()

        print("Loading %s..." % (rec_short_name))

        self.metrics[rec_short_name]={}
        for i,k in enumerate(experiment_constants.METRICS_K):

            result_file = open(self.get_file_name_metrics(base,k), 'r')
            
            self.metrics[rec_short_name][k]=[]
            for i,line in enumerate(result_file):
                obj=json.loads(line)
                self.metrics[rec_short_name][k].append(obj)


    def plot_acc_metrics(self,prefix_name='acc_met'):
        palette = plt.get_cmap(CMAP_NAME)
        ACC_METRICS = ['precision','recall','ndcg','ildg']
        for i,k in enumerate(experiment_constants.METRICS_K):
            metrics_mean=dict()
            for i,rec_using,metrics in zip(range(len(self.metrics)),self.metrics.keys(),self.metrics.values()):
                metrics=metrics[k]
                metrics_mean[rec_using]=defaultdict(float)
                for obj in metrics:
                    for key in obj:
                        if key in ACC_METRICS:
                            metrics_mean[rec_using][key]+=obj[key]
                for j,key in enumerate(metrics_mean[rec_using]):
                    metrics_mean[rec_using][key]/=len(metrics)
                    #print(f"{key}:{metrics_mean[rec_using][key]}")
            fig = plt.figure()
            ax=fig.add_subplot(111)
            barWidth= 1-len(self.metrics)/(1+len(self.metrics))
            N=len(ACC_METRICS)
            indexes=np.arange(N)
            i=0

            df = pd.DataFrame([],columns=ACC_METRICS)
            print(f"AT @{k}")
            for rec_using,rec_metrics in metrics_mean.items():
                # print(rec_metrics)
                df.loc[rec_using] = rec_metrics
                ax.bar(indexes+i*barWidth,rec_metrics.values(),barWidth,label=rec_using,color=palette(i))
                for ind in indexes:
                    ax.text(ind+i*barWidth-barWidth/2,0,"%.3f"%(list(rec_metrics.values())[ind]),fontsize=6)
                i+=1
                # ax.text(x=indexes+i*barWidth,
                #         y=np.zeros(N),
                #         s=['n']*N,
                #         fontsize=18)
            print(df.sort_values(by=ACC_METRICS[0],ascending=False))
            ax.set_xticks(np.arange(N+1)+barWidth*(((len(self.metrics))/2)-1)+barWidth/2)
            ax.legend(tuple(self.metrics.keys()))
            ax.set_xticklabels(ACC_METRICS)
            ax.set_title(f"at @{k}, {self.city}")
            fig.show()
            plt.show()
            timestamp = datetime.timestamp(datetime.now())
            fig.savefig(self.data_directory+f"{IMG}{prefix_name}_{self.city}_{str(k)}.png")


    def plot_gain_metrics(self,prefix_name='all_met_gain'):
        # palette = plt.get_cmap(CMAP_NAME)
        for i,k in enumerate(experiment_constants.METRICS_K):
            metrics_mean=dict()
            for i,rec_using,metrics in zip(range(len(self.metrics)),self.metrics.keys(),self.metrics.values()):
                metrics=metrics[k]
                
                metrics_mean[rec_using]=defaultdict(float)
                for obj in metrics:
                    for key in self.metrics_name:
                        metrics_mean[rec_using][key]+=obj[key]
                
                
                for j,key in enumerate(metrics_mean[rec_using]):
                    metrics_mean[rec_using][key]/=len(metrics)
                    #print(f"{key}:{metrics_mean[rec_using][key]}")
            # metrics_utility_score=dict()
            # for rec_using in metrics_mean:
            #     metrics_utility_score[rec_using]=dict()
            # for metric_name in self.metrics_name:
            #     max_metric_value=0
            #     min_metric_value=1
            #     for rec_using,rec_metrics in metrics_mean.items():
            #         max_metric_value=max(max_metric_value,rec_metrics[metric_name])
            #         min_metric_value=min(min_metric_value,rec_metrics[metric_name])
            #     for rec_using,rec_metrics in metrics_mean.items():
            #         metrics_utility_score[rec_using][metric_name]=(rec_metrics[metric_name]-min_metric_value)\
            #             /(max_metric_value-min_metric_value)

            reference_recommender = list(metrics_mean.keys())[0]
            reference_vals = np.array(list(metrics_mean.pop(reference_recommender).values()))
            fig = plt.figure()
            ax=fig.add_subplot(111)
            ax.grid(alpha=MPL_ALPHA)
            num_recs_plot = len(self.metrics)-1
            barWidth= 1-num_recs_plot/(1+num_recs_plot)
            N=len(self.metrics_name)
            indexes=np.arange(N)
            i=0
            styles = gen_bar_cycle(len(self.metrics))()
            for rec_using,rec_metrics in metrics_mean.items():
                print(f"{rec_using} at @{k}")
                print(rec_metrics)

                ax.bar(indexes+i*barWidth,-100+100*np.array(list(rec_metrics.values()))/reference_vals,barWidth,label=rec_using,**next(styles))
                #ax.bar(indexes[j]+i*barWidth,np.mean(list(rec_metrics.values())),barWidth,label=rec_using,color=palette(i))
                i+=1
            # reference_us_vals = metrics_utility_score.pop(reference_recommender)
            # reference_us_fvals = np.sum(list(reference_us_vals.values()))/len(reference_us_vals)
            # i=0
            # styles = gen_bar_cycle(len(self.metrics))()
            # for rec_using,utility_scores in metrics_utility_score.items():
            #     ax.bar(N+i*barWidth,
            #            -100+100*np.sum(list(utility_scores.values()))/len(utility_scores)/reference_us_fvals,
            #            barWidth,label=rec_using,**next(styles))
            #     i+=1

            #ax.set_xticks(np.arange(N+1)+barWidth*(np.floor((len(self.metrics))/2)-1)+barWidth/2)
            ax.set_xticks(np.arange(N+1)+barWidth*(((num_recs_plot)/2)-1)+barWidth/2)
            # ax.legend((p1[0], p2[0]), self.metrics_name)
            ax.legend(tuple(metrics_mean.keys()),bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3)
            # ax.legend(tuple(map(lambda name: METRICS_PRETTY[name],self.metrics.keys())))
            ax.set_xticklabels(list(map(lambda name: METRICS_PRETTY[name],self.metrics_name)))
            ax.set_ylabel("Gain after the reordering (%)")
            # ax.set_ylim(0,1)
            ax.set_xlim(-barWidth,len(self.metrics_name)-1+(num_recs_plot-1)*barWidth+barWidth)
            # ax.set_title(f"at @{k}, {self.city}")
            fig.show()
            plt.show()
            timestamp = datetime.timestamp(datetime.now())
            fig.savefig(self.data_directory+f"result/img/{prefix_name}_{self.city}_{str(k)}.png",bbox_inches="tight")

    def plot_bar_metrics(self,prefix_name='all_met'):
        palette = plt.get_cmap(CMAP_NAME)
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
                print(rec_metrics)
                ax.bar(indexes+i*barWidth,rec_metrics.values(),barWidth,label=rec_using,color=palette(i),edgecolor=BAR_EDGE_COLOR)
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
                ax.bar(N+i*barWidth,np.sum(list(utility_scores.values()))/len(utility_scores),barWidth,label=rec_using,color=palette(i),edgecolor=BAR_EDGE_COLOR)
                i+=1
            
                

            #ax.set_xticks(np.arange(N+1)+barWidth*(np.floor((len(self.metrics))/2)-1)+barWidth/2)
            ax.set_xticks(np.arange(N+1)+barWidth*(((len(self.metrics))/2)-1)+barWidth/2)
            # ax.legend((p1[0], p2[0]), self.metrics_name)
            ax.legend(tuple(self.metrics.keys()))
            # ax.legend(tuple(map(lambda name: METRICS_PRETTY[name],self.metrics.keys())))
            ax.set_xticklabels(list(map(lambda name: METRICS_PRETTY[name],self.metrics_name))+['MAUT'])
            # ax.set_title(f"at @{k}, {self.city}")
            ax.set_ylabel("Mean")
            fig.show()
            plt.show()
            timestamp = datetime.timestamp(datetime.now())
            fig.savefig(self.data_directory+f"result/img/{prefix_name}_{self.city}_{str(k)}.png")
            
                # ax.bar(indexes[j+1]+i*barWidth,np.mean(list(metrics_mean[rec_using].values())),barWidth,label=rec_using,color=palette(i))
    def test_data(self):
        self.invalid_uids = []
        self.message_start_section("TESTING DATA SET")
        has_some_error_global = False
        for i in self.all_uids:
            has_some_error = False
            test_size = len(self.ground_truth.get(i,[]))
            train_size = np.count_nonzero(self.training_matrix[i])
            if test_size == 0:
                print(f"user {i} with empty ground truth")
                has_some_error = True
                # remove from tests
                self.all_uids.remove(i)
                self.invalid_uids.append(i)
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
        x = 1
        r = 0.25
        l = []
        for i in np.append(np.arange(0, x, r),x):
            for j in np.append(np.arange(0, x, r),x):
                if not(i==0 and i!=j):
                    l.append((i,j))

        # for div_weight, div_geo_cat_weight in l:
        #     self.final_rec_parameters['div_weight'], self.final_rec_parameters['div_geo_cat_weight'] = div_weight, div_geo_cat_weight
        #     self.load_metrics(base=False,short_name=False)

        for i,K in enumerate(experiment_constants.METRICS_K):
            palette = plt.get_cmap(CMAP_NAME)
            fig = plt.figure(figsize=(8,8))
            ax=fig.add_subplot(111)
            ax.grid(alpha=MPL_ALPHA)
            plt.xticks(rotation=45)
            #K = max(experiment_constants.METRICS_K)
            #K = 10
            metrics_mean=dict()

            # There is some ways to do it more efficiently but i could not draw lines between points
            # this way is ultra slow but works
            styles = gen_line_cycle(len(self.metrics_name))()
            for i, metric_name in enumerate(self.metrics_name):
                metric_values = []
                for div_weight, div_geo_cat_weight in l:
                    self.final_rec_parameters['div_weight'], self.final_rec_parameters['div_geo_cat_weight'] = div_weight, div_geo_cat_weight
                    rec_using = self.get_final_rec_name()
                    metrics=self.metrics[rec_using][K]
                    metrics_mean[rec_using]=defaultdict(float)
                    for obj in metrics:
                        metrics_mean[rec_using][metric_name]+=obj[metric_name]
                    metrics_mean[rec_using][metric_name]/=len(metrics)
                    metric_values.append(metrics_mean[rec_using][metric_name])
                ax.plot(list(map(lambda pair: "%.2f-%.2f"%(pair[0],pair[1]),l)),metric_values,**next(styles))
            
            ax.legend(tuple(map(lambda name: METRICS_PRETTY[name],self.metrics_name)))
            ax.set_ylim(0,1)
            ax.set_ylabel("Mean Value")
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(13)
                tick.label.set_ha("right")
            timestamp = datetime.timestamp(datetime.now())
            fig.savefig(self.data_directory+IMG+f"geocat_parameters_{self.city}_{str(K)}.png")

    def plot_geopersonparameter(self):
        # self.load_base()
        # self.persongeocat_preprocess()

        fig=plt.figure()
        ax=fig.add_subplot(111)

        users_mean_walk = np.sort(self.pgeo_div_runner.users_mean_walk)
        mean_walk = self.pgeo_div_runner.mean_walk
        training_matrix = self.training_matrix

        uupper=np.ma.masked_where(users_mean_walk >= mean_walk, users_mean_walk)
        ulower=np.ma.masked_where(users_mean_walk < mean_walk, users_mean_walk)
        t=np.arange(0,training_matrix.shape[0],1)

        ax.plot(t, ulower, t, uupper,t,[mean_walk]*training_matrix.shape[0])

        ax.annotate('"Max" geographical diversification', xy=(1, 1), xytext=(1, mean_walk+0.1))
        ax.annotate("Users below: "+str(len(ulower[ulower.mask == False]))+", users above:"+str(len(uupper[uupper.mask == False])), xy=(0, 0), xytext=(0,0))

        ax.legend(('Users above upper bound', 'Users below upper bound', 'Upper bound distance'))
        ax.set_xlabel("User id")
        ax.set_ylabel("centroid mean distance")
        fig.savefig(self.data_directory+IMG+f'cmd_{self.city}.png')
        plt.show()

    def plot_geodivprop(self):

        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.plot(np.sort(self.geo_div_propensity))
        ax.set_xlabel("User id")
        ax.set_ylabel("Geographical diversification propensity")
        ax.set_title("Users geographical diversification propensity")
        fig.savefig(self.data_directory+IMG+f'gdp_{self.city}.png')
        plt.show()


    def plot_catdivprop(self):
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.plot(np.sort(self.cat_div_propensity))
        ax.set_xlabel("User id")
        ax.set_ylabel("Categorical diversification propensity")
        ax.set_title("Users categorical diversification propensity")
        fig.savefig(self.data_directory+IMG+f'cdp_{self.city}.png')
        plt.show()


    def plot_users_max_min_catprop(self):
        cat_div_prop = self.cat_div_propensity
        uid_cat_visits = self.pcat_div_runner.users_categories_visits

        uids=dict()
        uids['user min']=(np.argmin(cat_div_prop))
        uids['user median']=(np.argsort(cat_div_prop)[len(cat_div_prop)//2])
        uids['user max']=(np.argmax(cat_div_prop))
        for i, user_type in enumerate(uids):
            uid=uids[user_type]
            print(user_type)
            visits=list(uid_cat_visits[uid].values())

            fig=plt.figure()
            ax0, ax1=fig.subplots(1,2)

            ax0.plot(visits)
            ax0.set_xlabel("Category 'id'")
            ax0.set_ylabel("Visits")
            print(f"""std={np.std(visits)}
        skew={scipy.stats.skew(visits)}
        kurtosis={scipy.stats.kurtosis(visits)}
        Frequency hist skew={scipy.stats.skew(np.bincount(visits))}
        Frequency hist kurtosis={scipy.stats.kurtosis(visits)}
        different categories visited={len(visits)}
        """)

            ax1.hist(visits)
            ax1.set_xlabel("Visits")
            ax1.set_ylabel("Count")

            fig.savefig(self.data_directory+IMG+f'mmm_{self.city}_{user_type}.png')
            plt.show()

    def plot_relation_catdivprop_catvisits(self):
        cat_div_prop = np.array(self.cat_div_propensity)
        uid_cat_visits = self.pcat_div_runner.users_categories_visits

        argres=np.argsort(cat_div_prop)
        uid_cats=np.array(list(map(len,uid_cat_visits)))[argres]
        
        fig=plt.figure()
        ax=fig.subplots(1,1)
        ax.plot(uid_cats/np.max(uid_cats))
        ax.plot(cat_div_prop[argres])
        ax.annotate('Correlation='+f"{np.corrcoef(cat_div_prop[argres],uid_cats)[0,1]:.2f}", xy=(0, 1))
        ax.legend(["Number of categories visited","$\delta$ CatDivProp"])
        fig.savefig(self.data_directory+IMG+f'cdp_nc_{self.city}.png')
        plt.show()

    def plot_geocatdivprop(self):
        geo_div_prop = self.geo_div_propensity
        cat_div_prop = self.cat_div_propensity
        fig=plt.figure()
        ax = fig.subplots(1,1)
        t=np.arange(0,self.training_matrix.shape[0],1)
        ax.plot(t,geo_div_prop,t,cat_div_prop)
        ax.legend(("Geographical diversification","Categorical diversification"))
        ax.set_xlabel("User id")
        ax.set_ylabel("Diversification propensity")
        ax.set_title("Users cat and geo diversification propensity")
        fig.savefig(self.data_directory+IMG+f'gcdp_{self.city}.png')
        plt.show()
    def plot_personparameter(self):
        self.persongeocat_preprocess()
        # geo_div_prop = self.geo_div_propensity
        # cat_div_prop = self.cat_div_propensity
        training_matrix = self.training_matrix
        div_geo_cat_weight = self.div_geo_cat_weight
        fig=plt.figure()
        ax = fig.subplots(1,1)
        t=np.arange(0,training_matrix.shape[0],1)
        # ax.plot(t,geo_div_prop,t,cat_div_prop)
        # ax.legend(("Geographical diversification","Categorical diversification"))
        # ax.set_xlabel("User id")
        # ax.set_ylabel("Diversification propensity")
        # ax.set_title("Users cat and geo diversification propensity")
        plt.plot(np.sort(div_geo_cat_weight))
        plt.xlabel("Users")
        plt.ylabel("Value of $\\beta$")
        # plt.title("Value of $\\beta$ in the original formula $div_{geo-cat}(i,R)=\\beta\cdot div_{geo}(i,R)+(1-\\beta)\cdot div_{cat}(i,R)$")
        plt.title("%s, pgc %s-%s" % (self.city,self.final_rec_parameters['cat_div_method'],self.final_rec_parameters['geo_div_method']))

        plt.text(training_matrix.shape[0]/2,0.5,"median $\\beta$="+str(np.median(div_geo_cat_weight)))
        plt.savefig(self.data_directory+IMG+f'beta_{self.get_final_rec_name()}.png')
        plt.show()
    def plot_perfect_parameters(self):
        fin = open(self.data_directory+UTIL+f'parameter_{self.get_final_rec_name()}.pickle',"rb")
        self.perfect_parameter = pickle.load(fin)
        # print(self.perfect_parameter)
        vals = np.array([])
        for uid, val in self.perfect_parameter.items():
            vals = np.append(vals,val)
        plt.plot(np.sort(vals))
        # plt.plot(vals)
        # plt.hist(vals)
        plt.savefig(self.data_directory+IMG+f'perf_param_{self.get_final_rec_name()}.png')
        # plt.savefig(self.data_directory+IMG+f'perfect_{self.city}.png')


    def generate_general_user_data(self):
        preprocess_methods = [# [None,'walk']
                              # ,['poi_ild',None]
        ]
        final_rec = self.final_rec
        self.final_rec = 'persongeocat'
        div_geo_cat_weight = dict()
        methods_columns = []
        for cat_div_method in [None]+CatDivPropensity.METHODS:
            for geo_div_method in [None]+GeoDivPropensity.METHODS:
                if cat_div_method != geo_div_method and ([cat_div_method,geo_div_method] in preprocess_methods):
                    self.final_rec_parameters = {'cat_div_method': cat_div_method, 'geo_div_method': geo_div_method}
                    self.persongeocat_preprocess()
                    div_geo_cat_weight[(cat_div_method,geo_div_method)] = self.div_geo_cat_weight
                    methods_columns.append("%s-%s" % (cat_div_method,geo_div_method))
        df = pd.DataFrame([],
                          columns=[
                              'visits','visits_mean',
                              'visits_std','cats_visited',
                              'cats_visited_mean','cats_visited_std',
                              'num_friends'
                          ]
                          + methods_columns
        )
        # self.persongeocat_preprocess()
        # div_geo_cat_weight = self.div_geo_cat_weight
        for uid in self.all_uids:
            # beta = self.perfect_parameter[uid]
            visited_lids = self.training_matrix[uid].nonzero()[0]
            visits_mean = self.training_matrix[uid,visited_lids].mean()
            visits_std = self.training_matrix[uid,visited_lids].std()
            visits = self.training_matrix[uid,visited_lids].sum()
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
                    if cat_div_method != geo_div_method and ([cat_div_method,geo_div_method] in preprocess_methods):
                        methods_values.append(div_geo_cat_weight[(cat_div_method,geo_div_method)][uid])
            num_friends = len(self.social_relations[uid])
            df.loc[uid] = [visits,visits_mean,
                           visits_std,len(cats_visits),
                           cats_visits_mean,cats_visits_std,num_friends] + methods_values
        df = pd.concat([df,self.user_data],axis=1)

        # poly = PolynomialFeatures(degree=1,interaction_only=False)
        # res_poly = poly.fit_transform(df)
        # df_poly = pd.DataFrame(res_poly, columns=poly.get_feature_names(df.columns))
        df_poly = df
        
        
        self.final_rec = final_rec
        return df_poly

    def load_perfect(self):
        final_rec = self.final_rec
        self.final_rec = 'perfectpersongeocat'
        fin = open(self.data_directory+UTIL+f'parameter_{self.get_final_rec_name()}.pickle',"rb")
        self.perfect_parameter = pickle.load(fin)
        vals = np.array([])
        for uid, val in self.perfect_parameter.items():
            vals = np.append(vals,val)
        self.perfect_parameter = vals
        # self.perfect_parameter = pd.Series(self.perfect_parameter)
        self.final_rec = final_rec
    
    def print_correlation_perfectparam(self):
        self.base_rec = 'usg'

        self.load_perfect()
        self.perfect_parameter = pd.Series(self.perfect_parameter)

        df_poly=self.generate_general_user_data()

        correlations = df_poly.corrwith(self.perfect_parameter)
        print(correlations)
        print(correlations.idxmax(), correlations.max())

        X, y = df_poly, self.perfect_parameter
        lab_enc = preprocessing.LabelEncoder()
        y = lab_enc.fit_transform(y)

        X, y = imblearn.over_sampling.SMOTE().fit_resample(X, y)

        print(sorted(Counter(y).items()))

        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)



        for name,rp in regressor_predictors.items():
            rp.fit(X_train,y_train)
            print("=======",name,"=======")
            print("Train score", rp.score(X_train,y_train))
            print("Test score", rp.score(X_test,y_test))

        for name, cp in classifier_predictors.items():
            cp.fit(X_train,y_train)
            pred_cp = cp.predict(X_test)
            print("-------",name,"-------")
            print(classification_report(y_test, pred_cp,zero_division=0))
            print(confusion_matrix(y_test, pred_cp))


    def print_real_perfectparam(self):
        old_city = self.city
        bckp = vars(self).copy()

        df_test = self.generate_general_user_data()
        self.load_perfect()
        perfect_parameter_test = self.perfect_parameter
        perfect_parameter_train = np.array([])
        df = pd.DataFrame()
        for city in experiment_constants.CITIES:
            if city != old_city:
                self.city = city
                self.load_base()
                self.load_perfect()
                perfect_parameter_train = np.append(perfect_parameter_train,self.perfect_parameter)
                df = df.append(self.generate_general_user_data(), ignore_index=True)

        # print(perfect_parameter_train)
        # print(df)
        # print(df[df.isna().any(axis=1)])
        # print(df.shape,df_test.shape)
        # print(df_test.columns)

        for key, val in bckp.items():
            vars(self)[key] = val

        # print(len(self.perfect_parameter))
        # print(len(self.all_uids))
        X_train, X_test, y_train, y_test = df.to_numpy(),df_test.to_numpy(), perfect_parameter_train, perfect_parameter_test

        lab_enc = preprocessing.LabelEncoder()

        y_train = lab_enc.fit_transform(y_train)
        y_test = lab_enc.transform(y_test)

        prep_data = imblearn.over_sampling.SMOTE()
        # prep_data = imblearn.under_sampling.RandomUnderSampler(random_state=0)
        # prep_data = imblearn.under_sampling.NearMiss()

        X_train, y_train = prep_data.fit_resample(X_train, y_train)

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        for name,rp in regressor_predictors.items():
            rp.fit(X_train,y_train)
            print("=======",name,"=======")
            print("Train score", rp.score(X_train,y_train))
            print("Test score", rp.score(X_test,y_test))

        for name, cp in classifier_predictors.items():
            cp.fit(X_train,y_train)
            pred_cp = cp.predict(X_test)
            print("-------",name,"-------")
            print(classification_report(y_test, pred_cp,zero_division=0))
            print(confusion_matrix(y_test, pred_cp))


        self.base_rec = 'usg'

        self.load_perfect()
        self.perfect_parameter = pd.Series(self.perfect_parameter)

    def perfect_analysis(self):
        self.load_base()

        self.base_rec = 'usg'
        self.final_rec = 'perfectpersongeocat'

        fin = open(self.data_directory+UTIL+f'parameter_{self.get_final_rec_name()}.pickle',"rb")
        self.perfect_parameter = pickle.load(fin)

        vals = np.array([])
        for uid, val in self.perfect_parameter.items():
            vals = np.append(vals,val)
        args = np.argsort(vals)
        vals=vals[args]

        df = pd.DataFrame([])
        df['beta'] = vals

        self.load_metrics(base=False)
        df_metrics = pd.DataFrame(self.metrics[self.get_final_rec_short_name()][10])
        df_metrics = df_metrics.set_index('user_id')
        
        # print(df_metrics)
        df['precision'] = df_metrics['precision'][args]
        
        self.final_rec = 'persongeocat'
        self.final_rec_parameters = {'geo_div_method': 'walk', 'cat_div_method': None}
        self.persongeocat_preprocess()
        self.load_metrics(base=False)
        df_p_metrics = pd.DataFrame(self.metrics[self.get_final_rec_short_name()][10])
        df_p_metrics = df_p_metrics.set_index('user_id')
        df['p_precision'] = df_p_metrics['precision'][args]
        df['p_beta'] = self.div_geo_cat_weight[args]
        # df['beta_walk'] = self.div_geo_cat_weight
        # self.final_rec_parameters = {'geo_div_method': None, 'cat_div_method':'poi_ild'}
        # self.persongeocat_preprocess()
        # df['beta_poi_ild'] = self.div_geo_cat_weight
        # df= df.sort_values(by='beta').reset_index(drop=True)

        # print(df.head())
        print()
        print("Number of each beta with precision greather than 0:")
        print(df[df['precision']>0]['beta'].value_counts())
        print()
        # print(df[df['precision']>0].describe())
        df[['precision','p_precision','p_beta','beta']].plot()
        plt.savefig(self.data_directory+IMG+f'analysis_{self.get_final_rec_name()}.png')

    def print_latex_metrics_table(self,prefix_name='',references=[]):
        num_metrics = len(self.metrics_name)
        result_str = r"\begin{table}[]" + "\n"
        result_str += r"\begin{tabular}{" +'l|'+'l'*(num_metrics) + "}\n"
        # result_str += "\begin{tabular}{" + 'l'*(num_metrics+1) + "}\n"
        if not references:
            references = [list(self.metrics.keys())[0]]
        CITIES = [self.city]
        for city in CITIES:
            result_str += "\t&"+'\multicolumn{%d}{c}{%s}\\\\\n' % (num_metrics,CITIES_PRETTY[city])

            for i,k in enumerate(experiment_constants.METRICS_K):
                result_str += "\\hline \\textbf{Algorithm} & "+'& '.join(map(lambda x: "\\textbf{"+METRICS_PRETTY[x]+f"@{k}}}" ,self.metrics_name))+"\\\\\n"

                metrics_mean=dict()
                metrics_gain=dict()
                

                for i,rec_using,metrics in zip(range(len(self.metrics)),self.metrics.keys(),self.metrics.values()):
                    metrics=metrics[k]

                        # self.metrics[rec_using]
                    metrics_mean[rec_using]=defaultdict(float)
                    for obj in metrics:
                        for key in self.metrics_name:
                            metrics_mean[rec_using][key]+=obj[key]

                    for j,key in enumerate(metrics_mean[rec_using]):
                        metrics_mean[rec_using][key]/=len(metrics)

                for i,rec_using,metrics in zip(range(len(self.metrics)),self.metrics.keys(),self.metrics.values()):
                    metrics_k=metrics[k]
                    if rec_using not in references:
                        metrics_gain[rec_using] = dict()
                        for metric_name in self.metrics_name:
                            statistic, pvalue = scipy.stats.wilcoxon(
                                [ms[metric_name] for ms in base_metrics[k]],
                                [ms[metric_name] for ms in metrics_k],
                            )
                            if pvalue > 0.05:
                                metrics_gain[rec_using][metric_name] = r'\textcolor[rgb]{0.7,0.7,0.0}{$\bullet$}'
                            else:
                                if metrics_mean[rec_using][metric_name] < metrics_mean[reference_name][metric_name]:
                                    metrics_gain[rec_using][metric_name] = r'\textcolor[rgb]{0.7,00,00}{$\blacktriangledown$}'
                                elif metrics_mean[rec_using][metric_name] > metrics_mean[reference_name][metric_name]:
                                    metrics_gain[rec_using][metric_name] = r'\textcolor[rgb]{00,0.45,0.10}{$\blacktriangle$}'
                                else:
                                    metrics_gain[rec_using][metric_name] = r'\textcolor[rgb]{0.7,0.7,0.0}{$\bullet$}'
                    else:
                        reference_name = rec_using
                        base_metrics = metrics
                        metrics_gain[rec_using] = dict()
                        for metric_name in self.metrics_name:
                            metrics_gain[rec_using][metric_name] = ''

                for rec_using,rec_metrics in metrics_mean.items():
                    gain = metrics_gain[rec_using]
                    result_str += rec_using +' &'+ '& '.join(map(lambda x: "%.4f%s"%(x[0],x[1]) ,zip(rec_metrics.values(),gain.values()) ))+"\\\\\n"

        result_str += "\\end{tabular}\n"
        result_str += "\\end{table}\n"
        result_str = LATEX_HEADER + result_str
        result_str += LATEX_FOOT
        fout = open(self.data_directory+UTIL+'_'.join([prefix_name]+CITIES)+'.tex', 'w')
        fout.write(result_str)
        fout.close()

        # plt.figure(figsize=(4,5))
        # ax=plt.subplot2grid((1,1),(0,0))
        # ax.text(0.5,0.8,result_str,ha="center",va="center",transform=ax.transAxes)
        # plt.savefig(self.data_directory+IMG+'_'.join([prefix_name]+CITIES)+'.png',bbox_inches="tight")

    def plot_bar_exclusive_metrics(self,prefix_name='all_met',ncol=3):
        palette = plt.get_cmap(CMAP_NAME)
        for i,k in enumerate(experiment_constants.METRICS_K):
            metrics_mean=dict()
            for i,rec_using,metrics in zip(range(len(self.metrics)),self.metrics.keys(),self.metrics.values()):
                metrics=metrics[k]
                
                metrics_mean[rec_using]=defaultdict(float)
                for obj in metrics:
                    for key in self.metrics_name:
                        metrics_mean[rec_using][key]+=obj[key]
                
                for j,key in enumerate(metrics_mean[rec_using]):
                    metrics_mean[rec_using][key]/=len(metrics)
                    #print(f"{key}:{metrics_mean[rec_using][key]}")
            fig = plt.figure()
            ax=fig.add_subplot(111)
            ax.grid(alpha=MPL_ALPHA)
            barWidth= 1-len(self.metrics)/(1+len(self.metrics))
            N=len(self.metrics_name)
            indexes=np.arange(N)
            i=0
            styles = gen_bar_cycle(len(self.metrics))()
            for rec_using,rec_metrics in metrics_mean.items():
                print(f"{rec_using} at @{k}")
                print(rec_metrics)
                ax.bar(indexes+i*barWidth,rec_metrics.values(),barWidth,label=rec_using,**next(styles))
                #ax.bar(indexes[j]+i*barWidth,np.mean(list(rec_metrics.values())),barWidth,label=rec_using,color=palette(i))
                i+=1

            #ax.set_xticks(np.arange(N+1)+barWidth*(np.floor((len(self.metrics))/2)-1)+barWidth/2)
            ax.set_xticks(np.arange(N+1)+barWidth*(((len(self.metrics))/2)-1)+barWidth/2)
            # ax.legend((p1[0], p2[0]), self.metrics_name)
            ax.legend(tuple(self.metrics.keys()),bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                      mode="expand", borderaxespad=0, ncol=ncol)
            # ax.legend(tuple(map(lambda name: METRICS_PRETTY[name],self.metrics.keys())))
            ax.set_xticklabels(list(map(lambda name: METRICS_PRETTY[name],self.metrics_name)))
            # ax.set_title(f"at @{k}, {self.city}")
            ax.set_ylabel("Mean Value")
            ax.set_ylim(0,1)
            ax.set_xlim(-barWidth,len(self.metrics_name)-1+(len(self.metrics)-1)*barWidth+barWidth)
            fig.show()
            plt.show()
            timestamp = datetime.timestamp(datetime.now())
            fig.savefig(self.data_directory+f"result/img/{prefix_name}_{self.city}_{str(k)}.png",bbox_inches="tight")

    def plot_maut(self,prefix_name='maut',ncol=3,print_times=False):
        # palette = plt.get_cmap(CMAP_NAME)
        fig = plt.figure()
        ax=fig.add_subplot(111)

        N=len(experiment_constants.METRICS_K)
        barWidth=1-len(self.metrics)/(1+len(self.metrics))
        indexes=np.arange(N)

        for count,k in enumerate(experiment_constants.METRICS_K):
            metrics_mean=dict()
            for i,rec_using,metrics in zip(range(len(self.metrics)),self.metrics.keys(),self.metrics.values()):
                metrics=metrics[k]

                metrics_mean[rec_using]=defaultdict(float)
                for obj in metrics:
                    for key in self.metrics_name:
                        metrics_mean[rec_using][key]+=obj[key]


                for j,key in enumerate(metrics_mean[rec_using]):
                    metrics_mean[rec_using][key]/=len(metrics)
                    #print(f"{key}:{metrics_mean[rec_using][key]}")
            styles = gen_bar_cycle(len(self.metrics))()
            # ax.spines['top'].set_visible(False)
            # ax.spines['right'].set_visible(False)
            # ax.spines['bottom'].set_visible(False)
            # ax.spines['left'].set_visible(False)

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
                val = np.sum(list(utility_scores.values()))/len(utility_scores)
                ax.bar(count+i*barWidth,val,barWidth,**next(styles),label=rec_using)
                    # ax.bar(count+i*barWidth,val,barWidth,**GEOCAT_BAR_STYLE,label=rec_using)
                ax.text(count+i*barWidth-barWidth/2+barWidth*0.1,val+val*0.05,"%.2f"%(val),rotation=90)
                i+=1
            #ax.set_xticks(np.arange(N+1)+barWidth*(np.floor((len(self.metrics))/2)-1)+barWidth/2)
        # ax.set_xticks(np.arange(N+1)+barWidth*len(metrics_utility_score)/2+barWidth/2)

        ax.set_xticks(np.arange(N+1)+barWidth*(((len(self.metrics))/2)-1)+barWidth/2)
        # ax.legend((p1[0], p2[0]), self.metrics_name)
        ax.legend(tuple(self.metrics.keys()),bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                  mode="expand", borderaxespad=0, ncol=ncol)
        # ax.legend(tuple(map(lambda name: METRICS_PRETTY[name],self.metrics.keys())))
        ax.set_xticklabels(['MAUT@5','MAUT@10','MAUT@20'])
        # ax.set_title(f"at @{k}, {self.city}")
        ax.set_ylabel("Mean Value")
        ax.set_ylim(0,1)
        ax.set_xlim(-barWidth,len(experiment_constants.METRICS_K)-1+(len(self.metrics)-1)*barWidth+barWidth)

        if print_times:
            textstr = '\n'.join((
                'Tabu search: xxx',
                'Particle swarm: 6.35',
                'Greedy: ',
            ))
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

            # place a text box in upper left in axes coords
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                    verticalalignment='top', bbox=props)
        fig.show()
        plt.show()

        timestamp = datetime.timestamp(datetime.now())
        fig.savefig(self.data_directory+f"result/img/{prefix_name}_{self.city}.png",bbox_inches="tight")

    def print_ild_gc_correlation(self,metrics=['ild','gc']):
        rec = list(self.metrics.keys())[0]
        metrics_ks = list(self.metrics.values())[0]

        for k, metrics_k in metrics_ks.items():
            print("%s@%d correlation"%(rec,k))
            df_p_metrics = pd.DataFrame(metrics_k)
            df_p_metrics = df_p_metrics.set_index('user_id')
            print(df_p_metrics[metrics].corr())

    def plot_geocat_parameters_maut(self):
        x = 1
        r = 0.25
        l = []
        for i in np.append(np.arange(0, x, r),x):
            for j in np.append(np.arange(0, x, r),x):
                if not(i==0 and i!=j):
                    l.append((i,j))

        for i,k in enumerate(experiment_constants.METRICS_K):
            palette = plt.get_cmap(CMAP_NAME)
            fig = plt.figure(figsize=(8,8))
            ax=fig.add_subplot(111)
            ax.grid(alpha=MPL_ALPHA)
            plt.xticks(rotation=45)
            #K = max(experiment_constants.METRICS_K)
            #K = 10
            metrics_mean=dict()
            for i,rec_using,metrics in zip(range(len(self.metrics)),self.metrics.keys(),self.metrics.values()):
                metrics=metrics[k]

                metrics_mean[rec_using]=defaultdict(float)
                for obj in metrics:
                    for key in self.metrics_name:
                        metrics_mean[rec_using][key]+=obj[key]


                for j,key in enumerate(metrics_mean[rec_using]):
                    metrics_mean[rec_using][key]/=len(metrics)

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
            mauts = dict()
            for rec_using,utility_scores in metrics_utility_score.items():
                mauts[rec_using] = np.sum(list(utility_scores.values()))/len(utility_scores)

            styles = gen_bar_cycle(len(self.metrics_name))()

            ax.bar(list(map(lambda pair: "%.2f-%.2f"%(pair[0],pair[1]),l)),
                   list(mauts.values()),
                   **next(styles))
            
            ax.set_ylim(0,1)
            ax.set_ylabel("MAUT")
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(13)
                tick.label.set_ha("right")
            timestamp = datetime.timestamp(datetime.now())
            fig.savefig(self.data_directory+IMG+f"geocat_parameters_maut_{self.city}_{str(k)}.png")

    def plot_geocat_parameters(self):
        x = 1
        r = 0.25
        l = []
        for i in np.append(np.arange(0, x, r),x):
            for j in np.append(np.arange(0, x, r),x):
                if not(i==0 and i!=j):
                    l.append((i,j))

        for div_weight, div_geo_cat_weight in l:
            self.final_rec_parameters['div_weight'], self.final_rec_parameters['div_geo_cat_weight'] = div_weight, div_geo_cat_weight
            self.load_metrics(base=False,short_name=False)
        self.plot_geocat_parameters_metrics()
        self.plot_geocat_parameters_maut()

    def plot_geocat_div_cat_weights_maut(self,l):
        for i,k in enumerate(experiment_constants.METRICS_K):
            palette = plt.get_cmap(CMAP_NAME)
            fig = plt.figure(figsize=(8,8))
            ax=fig.add_subplot(111)
            # ax.grid(alpha=MPL_ALPHA)
            plt.xticks(rotation=90)
            #K = max(experiment_constants.METRICS_K)
            #K = 10
            metrics_mean=dict()
            for i,rec_using,metrics in zip(range(len(self.metrics)),self.metrics.keys(),self.metrics.values()):
                metrics=metrics[k]

                metrics_mean[rec_using]=defaultdict(float)
                for obj in metrics:
                    for key in self.metrics_name:
                        metrics_mean[rec_using][key]+=obj[key]


                for j,key in enumerate(metrics_mean[rec_using]):
                    metrics_mean[rec_using][key]/=len(metrics)

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
            mauts = dict()
            for rec_using,utility_scores in metrics_utility_score.items():
                mauts[rec_using] = np.sum(list(utility_scores.values()))/len(utility_scores)

            styles = gen_line_cycle(1)()

            print(l)
            print(list(mauts.values()))
            ax.plot(l,
                   list(mauts.values()),
                   **next(styles),
            )
            
            ax.set_ylim(0,1)
            ax.set_xlim(0,1)
            ax.set_ylabel("MAUT")
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(11)
                tick.label.set_ha("right")
            timestamp = datetime.timestamp(datetime.now())
            fig.savefig(self.data_directory+IMG+f"div_cat_weights_maut_{self.city}_{str(k)}.png")

    def plot_geocat_div_cat_weights_metrics(self,l):
        for i,K in enumerate(experiment_constants.METRICS_K):
            palette = plt.get_cmap(CMAP_NAME)
            fig = plt.figure(figsize=(8,8))
            ax=fig.add_subplot(111)
            # ax.grid(alpha=MPL_ALPHA)
            plt.xticks(rotation=45)
            #K = max(experiment_constants.METRICS_K)
            #K = 10
            metrics_mean=dict()

            # There is some ways to do it more efficiently but i could not draw lines between points
            # this way is ultra slow but works
            styles = gen_line_cycle(len(self.metrics_name))()
            for i, metric_name in enumerate(self.metrics_name):
                metric_values = []
                for div_cat_weight in l:
                    self.final_rec_parameters['div_cat_weight'] = div_cat_weight
                    rec_using = self.get_final_rec_name()
                    metrics=self.metrics[rec_using][K]
                    metrics_mean[rec_using]=defaultdict(float)
                    for obj in metrics:
                        metrics_mean[rec_using][metric_name]+=obj[metric_name]
                    metrics_mean[rec_using][metric_name]/=len(metrics)
                    metric_values.append(metrics_mean[rec_using][metric_name])
                ax.plot(l,metric_values,**next(styles))
            
            ax.legend(tuple(map(lambda name: METRICS_PRETTY[name],self.metrics_name)))
            ax.set_ylim(0,1)
            ax.set_ylabel("Mean Value")
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(13)
                tick.label.set_ha("right")
            timestamp = datetime.timestamp(datetime.now())
            fig.savefig(self.data_directory+IMG+f"div_cat_weights_metrics_{self.city}_{str(K)}.png")

    def plot_geocat_div_cat_weights(self):

        x = 1
        r = 0.1
        l = list(np.around(np.append(np.arange(0, x, r),x),decimals=2))
        for i in l:
            self.final_rec_parameters['div_cat_weight']=i
            self.load_metrics(base=False,short_name=False)
        self.plot_geocat_div_cat_weights_metrics(l)
        self.plot_geocat_div_cat_weights_maut(l)



    # def plot_geo_cat_methods_vs(self):
    #     questions = [
    #         inquirer.Checkbox('geo_div_method',
    #                           message="Geographical diversification method",
    #                           choices=GeoDivPropensity.METHODS,
    #         )]

    #     answers = inquirer.prompt(questions)
    #     geo_div_methods = answers['geo_div_method']

    #     self.pgeo_div_runner = GeoDivPropensity.getInstance(self.training_matrix, self.poi_coos,
    #                                                         self.poi_cats,self.undirected_category_tree,
    #                                                         '')
    #     for i, geo_div_method1 in enumerate(geo_div_methods):
    #         self.pgeo_div_runner.geo_div_method = geo_div_method1
    #         geo_div_propensity1 = self.pgeo_div_runner.compute_div_propensity()
    #         for j, geo_div_method2 in enumerate(geo_div_methods):
    #             if j < i:
    #                 self.pgeo_div_runner.geo_div_method = geo_div_method2
    #                 geo_div_propensity2 = self.pgeo_div_runner.compute_div_propensity()
    #                 args = np.argsort(geo_div_propensity1)
    #                 fig = plt.figure(figsize=(8,8))
    #                 ax = fig.add_subplot(111)
    #                 ax.plot(geo_div_propensity1)
    #                 ax.plot(geo_div_propensity2)
    #                 ax.annotate("Correlation: %f"%(scipy.stats.pearsonr(geo_div_method1,geo_div_method2)[0]), xy=(0, 0), xytext=(0,0),zorder = 21)
    #                 fig.savefig(self.data_directory+IMG+f"{self.city}_{geo_div_method1}_{geo_div_method2}_corr.png")
    
    def plot_geo_cat_methods(self):
        questions = [
        inquirer.Checkbox('geo_div_method',
                            message="Geographical diversification method",
                            choices=GeoDivPropensity.METHODS,
                            ),
        inquirer.Checkbox('cat_div_method',
                            message="Categorical diversification method",
                            choices=CatDivPropensity.METHODS,
                            ),
        ]
        answers = inquirer.prompt(questions)
        geo_div_methods, cat_div_methods = answers['geo_div_method'], answers['cat_div_method']
        df = pd.DataFrame([])
        geo_div_propensities = dict()

        for geo_div_method in geo_div_methods:
            self.pgeo_div_runner = GeoDivPropensity.getInstance(self.training_matrix, self.poi_coos,
                                                                self.poi_cats,self.undirected_category_tree,
                                                                geo_div_method)
            geo_div_propensities[geo_div_method] = self.pgeo_div_runner.compute_div_propensity()

        cat_div_propensities = dict()
        for cat_div_method in cat_div_methods:
            self.pcat_div_runner = CatDivPropensity.getInstance(
                self.training_matrix,
                self.undirected_category_tree,
                cat_div_method,
                self.poi_cats)
            cat_div_propensities[cat_div_method]=self.pcat_div_runner.compute_div_propensity()

        # print("Geographical diversification propensity methods correlation")
        # print(pd.DataFrame(geo_div_propensties).corr())

        # print("Categorical diversification propensity methods correlation")
        # print(pd.DataFrame(cat_div_propensties).corr())
        print(pd.concat([pd.DataFrame(geo_div_propensities),pd.DataFrame(cat_div_propensities)],axis=1).corr())
        
        for geo_div_method in geo_div_methods:
            for cat_div_method in cat_div_methods:
                geo_div_propensity = geo_div_propensities[geo_div_method]
                cat_div_propensity = cat_div_propensities[cat_div_method]
                fig = plt.figure(figsize=(8,8))
                ax = fig.add_subplot(111)
                # ax.set_ylim(0,1)
                cat_median = np.median(cat_div_propensity)
                geo_median = np.median(geo_div_propensity)
                groups = dict()
                
                groups['geo_preference'] = (cat_div_propensity <= cat_median) & (geo_div_propensity > geo_median)

                groups['no_preference'] = ((cat_div_propensity <= cat_median) & (geo_div_propensity <= geo_median)) | ((cat_div_propensity >= cat_median) & (geo_div_propensity >= geo_median))

                groups['cat_preference'] = (cat_div_propensity > cat_median) & (geo_div_propensity <= geo_median)

                assert(np.max(groups['no_preference'] & groups['cat_preference'] & groups['geo_preference']) == 0)
                color = 0.8
                for group, mask in groups.items():
                    # print(mask)
                    # print(len(cat_div_propensity[mask]))
                    # print(len(geo_div_propensity[mask]))
                    ax.scatter(cat_div_propensity[mask],geo_div_propensity[mask],color=str(color))
                    color -= 0.2
                # ax.plot([cat_median]*2,[0,1],color='k')
                # ax.plot([0,1],[geo_median]*2,color='k')
                # ax.plot([cat_median]*2,ax.get_ylim(),color='k')
                # ax.plot(ax.get_ylim(),[geo_median]*2,color='k')
                # ax.set_xlim([min(cat_div_propensity),max(cat_div_propensity)])
                # ax.set_ylim([min(geo_div_propensity),max(geo_div_propensity)])
                ax.set_xlabel('Categorical method ('+CatDivPropensity.CAT_DIV_PROPENSITY_METHODS_PRETTY_NAME[cat_div_method]+')')
                ax.set_ylabel('Geographic method ('+GeoDivPropensity.GEO_DIV_PROPENSITY_METHODS_PRETTY_NAME[geo_div_method]+')')
                # ax.set_xlim(min(np.min(cat_div_propensity),0),max(np.max(cat_div_propensity),1))
                # ax.set_ylim(min(np.min(geo_div_propensity),0),max(np.max(geo_div_propensity),1))

                ax.set_title("Correlation: %f"%(scipy.stats.pearsonr(cat_div_propensity,geo_div_propensity)[0]))
                ax.legend((f"Geographical preference ({np.count_nonzero(groups['geo_preference'])} people's)",
                           f"No preference ({np.count_nonzero(groups['no_preference'])} people's)",
                           f"Categorical preference ({np.count_nonzero(groups['cat_preference'])} people's)"))
                ax.plot([cat_median]*2,[min(geo_div_propensity),max(geo_div_propensity)],color='k')
                ax.plot([min(cat_div_propensity),max(cat_div_propensity)],[geo_median]*2,color='k')

                fig.savefig(self.data_directory+IMG+f"{self.city}_{geo_div_method}_{cat_div_method}.png")
        
