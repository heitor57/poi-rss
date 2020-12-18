# import os
# import sys
# module_path = os.path.abspath(os.path.join('.'))
# if module_path not in sys.path:
#     sys.path.append(module_path)
#print(sys.path)
LANG = 'en'
from enum import Enum
class NameType(Enum):
    SHORT = 1
    PRETTY = 2
    BASE_PRETTY = 3
    CITY_BASE_PRETTY = 4
    FULL = 5

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

LATEX_HEADER = ''
LATEX_FOOT = ''

# bullet_str = r'\textcolor[rgb]{0.25,0.25,0.25}{$\bullet$}'
# triangle_up_str = r'\textcolor[rgb]{0.0,0.0,0.0}{$\blacktriangle$}'
# triangle_down_str = r'\textcolor[rgb]{0.5,0.5,0.5}{$\blacktriangledown$}'
bullet_str = r'\textcolor[rgb]{0.7,0.7,0.0}{$\bullet$}'
triangle_up_str = r'\textcolor[rgb]{00,0.45,0.10}{$\blacktriangle$}'
triangle_down_str = r'\textcolor[rgb]{0.7,00,00}{$\blacktriangledown$}'


from abc import ABC, abstractmethod
from collections import defaultdict, OrderedDict, Counter
import pickle
from concurrent.futures import ProcessPoolExecutor
import json
import time
from datetime import datetime
import itertools
import multiprocessing
from collections import Counter
import os
import math

import inquirer
import numpy as np
from tqdm import tqdm
from scipy.stats import describe
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from mpl_toolkits.mplot3d import Axes3D
from cycler import cycler
# PALETTE = plt.get_cmap('Greys')
# monochrome = (cycler('color', ['k']) * cycler('linestyle', ['-.', '--','-', ':', ]) * cycler('marker', [',','.','^']))
# plt.rcParams['font.size'] = 11
# plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.framealpha'] = 1.0
# plt.rcParams['legend.shadow'] = True
plt.rcParams['legend.fancybox'] = False
# plt.rcParams['axes.grid'] = True
# plt.rcParams['axes.prop_cycle'] = monochrome
# plt.rcParams['axes.spines.top'] = False
# plt.rcParams['axes.spines.right'] = False

BAR_EDGE_COLOR = 'black'

def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

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

# ['#4477AA','#66CCEE','#228833','#CCBB44','#EE6677','#AA3377','#BBBBBB']
# my color scheme
# ['#9595ff','#2a913e','#ffb2b2','#b5b355','#11166c','#ecd9c6','#939393']
MY_COLOR_SCHEME = ['#939393','#9595ff','#2a913e','#b5b355','#11166c','#ffb2b2','#ecd9c6']
def brightness(color):
    r = int(color[1:3],16)
    b = int(color[3:5],16)
    g = int(color[5:7],16)
    return 0.2126*r + 0.0722*b + 0.7152*g

def ord_scheme_brightness(color_scheme):
    colors_brightness = list(map(brightness,MY_COLOR_SCHEME))
    color_scheme = [x for y, x in sorted(zip(colors_brightness, color_scheme))]
    return color_scheme
def get_my_color_scheme(num=None,ord_by_brightness=False,inverse_order=True):
    global MY_COLOR_SCHEME
    my_color_scheme = MY_COLOR_SCHEME.copy()
    if ord_by_brightness:
        my_color_scheme=ord_scheme_brightness(my_color_scheme)

    if inverse_order:
        my_color_scheme = my_color_scheme[-num:]

    if num:
        return my_color_scheme[:num]
    return my_color_scheme
            

def gen_bar_cycle(num=6,ord_by_brightness=False,inverse_order=True):
    my_color_scheme = get_my_color_scheme(num,ord_by_brightness,inverse_order)

    arrange = np.linspace(0,1,num)
    # hatch_cycler = cycler('hatch', ['///', '--', '...','\///', 'xxx', '\\\\','None','*'][:num])
    base_cycler = cycler('zorder', [10])# *cycler('edgecolor',[BAR_EDGE_COLOR])
    # base_cycler = cycler('zorder', [10])*cycler('edgecolor',[adjust_lightness(color,amount=0.3) for color in ['#9595ff','#2a913e','#ffb2b2','#b5b355','#11166c','#ecd9c6','#939393'][:num]])

    # color_cycler = cycler('color', reversed(list(map(str,arrange))))
    color_cycler = cycler('color', my_color_scheme)+cycler('edgecolor',[adjust_lightness(color,amount=0.3) for color in my_color_scheme])
    # color_cycler = cycler('color', reversed([plt.get_cmap('Dark2')(i) for i in arrange]))
    linewidth_cycler = cycler('linewidth', [1])
    bar_cycle = (# hatch_cycler +
                 color_cycler)*base_cycler*linewidth_cycler
    return bar_cycle
GEOCAT_BAR_STYLE = {'color': 'k', 'zorder': 10,'edgecolor':BAR_EDGE_COLOR}

def int_what_ordinal(num):
    num = str(num)
    if len(num) > 2:
        end_digits = int(num) % 100
    else:
        end_digits = int(num) % 10
    if end_digits == 1:
        return "st"
    if end_digits == 2:
        return "nd"
    if end_digits == 3:
        return "rd"
    else:
        return "th"

def sec_to_hm(s):
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f'{h:d}h{m:02d}m'

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
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
from sklearn import preprocessing
from sklearn import svm
from sklearn.preprocessing import PolynomialFeatures
from sklearn import neighbors
from sklearn.cluster import DBSCAN
import imblearn
import imblearn.datasets
import imblearn.over_sampling
import imblearn.under_sampling
import scipy.stats
from matplotlib.legend import Legend

import cat_utils
from usg.UserBasedCF import UserBasedCF
from usg.FriendBasedCF import FriendBasedCF
from usg.PowerLaw import PowerLaw
import geocat.objfunc as gcobjfunc
from pgc.GeoDivPropensity import GeoDivPropensity
from pgc.CatDivPropensity import CatDivPropensity
from GeoMF import GeoMF
from GeoDiv2020 import GeoDiv2020
from constants import experiment_constants, METRICS_PRETTY, RECS_PRETTY, CITIES_PRETTY, HEURISTICS_PRETTY, GROUP_ID, CITIES_BEST_PARAMETERS
import metrics
from geocat.Binomial import Binomial
from geocat.Pm2 import Pm2
from parallel_util import run_parallel
from geosoca.AdaptiveKernelDensityEstimation import AdaptiveKernelDensityEstimation
from geosoca.SocialCorrelation import SocialCorrelation
from geosoca.CategoricalCorrelation import CategoricalCorrelation
from Text import Text
from Arrow3D import Arrow3D
from geocat.gc import gc_diversifier
import geo_utils
from geocat.random import random_diversifier
from HandlerSquare import HandlerSquare
CMAP_NAME = 'viridis'

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
    PARAMETERS_BY_CITY = False
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
            "perfectpgeocat": self.perfectpersongeocat,
            "pdpgeocat": self.pdpgeocat,
            "gc": self.gc,
            "random": self.random,
        }
        # self.BASE_RECOMMENDERS_PARAMETERS = {
        #     "mostpopular": [],
        #     "usg": ['alpha','beta','eta']
        # }
        # self.FINAL_RECOMMENDERS_PARAMETERS = {
        #     "geocat": ['div_weight','div_geo_cat_weight'],
        #     "persongeocat": ['div_weight']
        # }
        self.city = city
        self.base_rec = base_rec
        self.final_rec = final_rec

        self.base_rec_parameters = base_rec_parameters
        self.final_rec_parameters = final_rec_parameters


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
        self.groups_epc = {}
        self.metrics_name = ['precision', 'recall', 'gc', 'ild','pr','epc']
        
        self.except_final_rec = except_final_rec
        self.welcome_message()
        self.CHKS = 50 # chunk size for process pool executor
        self.CHKSL = 100# chunk size for process pool executor largest
        self.cache = defaultdict(dict)
        self.metrics_cities = dict()

        self.show_heuristic = False
        self.persons_plot_special_case = False
        self.k_fold = None
        self.fold = None
        self.train_size = None
        self.recs_user_final_predicted_lid = {}
        self.recs_user_final_predicted_score = {}
        self.recs_user_base_predicted_lid = {}
        self.recs_user_base_predicted_score = {}
        

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
        if self.PARAMETERS_BY_CITY:
            final_parameters = CITIES_BEST_PARAMETERS[self.city][self.final_rec]
        else:
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
            "geomf": {'K': 100, 'delta': 50, 'gamma': 0.01, 'epsilon': 10, 'lambda_': 10, 'max_iters': 7, 'grid_distance':3.0},
            "mostpopular": {},
            # "usg": {'alpha': 0.1, 'beta': 0.1, 'eta': 0.05},
            "usg": {'alpha': 0, 'beta': 0.2, 'eta': 0},
            "geosoca": {'alpha': 0.3},
        }

    @classmethod
    def get_final_parameters(cls):
        return  {
            "geodiv2020": {'div_weight':0.5},
            "geocat": {'div_weight':0.75,'div_geo_cat_weight':0.25, 'heuristic': 'local_max', 'obj_func': 'cat_weight', 'div_cat_weight': 0.05},
            "persongeocat": {'div_weight':0.75,'cat_div_method': 'inv_num_cat', 'geo_div_method': 'walk',
                             'obj_func': 'cat_weight', 'div_cat_weight':0.05, 'bins': None,
                             'norm_method': 'default','funnel':None},
            "geodiv": {'div_weight':0.5},
            "ld": {'div_weight':0.25},
            "binomial": {'alpha': 0.5, 'div_weight': 0.75},
            "pm2": {'div_weight': 1},
            "perfectpgeocat": {'k': 10,'interval': 2,'div_weight': 0.75,'div_cat_weight': 0.05, 'train_size': None},
            "pdpgeocat": {'k': 10,'interval': 2,'div_geo_cat_weight': 0.75},
            "gc": {'div_weight': 0.8},
            "random": {'div_weight': 1},
        }

    def get_base_rec_name(self):
        list_parameters=list(map(str,dict_to_list(self.base_rec_parameters)))
        string="_" if len(list_parameters)>0 else ""

        if self.k_fold != None:
            k_fold = f'{self.k_fold}_{self.fold}_'
        else:
            k_fold = ''

        if self.train_size != None:
            train_size = f'{self.train_size}_'
        else:
            train_size = ''

        return f"{train_size}{k_fold}{self.city}_{self.base_rec}_{self.base_rec_list_size}"+\
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
            return f"{self.get_base_rec_short_name()}_{self.final_rec}_{self.final_rec_parameters['div_geo_cat_weight']}_{self.final_rec_parameters['div_cat_weight']}"
        elif self.final_rec == 'persongeocat':
            string = f"{self.get_base_rec_short_name()}_{self.final_rec}"
            if self.persons_plot_special_case:
                string = ''
            # if self.final_rec_parameters['cat_div_method']:
            string += f"_{self.final_rec_parameters['cat_div_method']}"
            if self.persons_plot_special_case:
                string = string[1:]
            # if self.final_rec_parameters['geo_div_method']:
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
        if self.final_rec == 'geocat' and self.final_rec_parameters['div_geo_cat_weight']==1:
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

    def load_base(self,user_data=False,test_data=True):
        CITY = self.city

        print(f"{CITY} city base loading")

        if self.train_size != None:
            self.training_matrix = pickle.load(open(self.data_directory+UTIL+f'train_val_{self.get_train_validation_name()}.pickle','rb'))
            self.ground_truth = pickle.load(open(self.data_directory+UTIL+f'test_val_{self.get_train_validation_name()}.pickle','rb'))

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
        self.dict_alias_title, self.category_tree, self.dict_alias_depth = cat_utils.cat_structs_igraph(
            self.data_directory+"categories.json")
        # self.undirected_category_tree = self.category_tree.to_undirected()
        self.undirected_category_tree = self.category_tree.shortest_paths()

        cats = self.category_tree.vs['name']
        self.cat_num = len(cats)
        cats_to_int = dict()
        for i, cat in enumerate(cats):
            cats_to_int[cat] = i
        self.poi_cats = {poi_id: [cats_to_int[cat] for cat in cats] for poi_id, cats in self.poi_cats.items()}

        # Training matrix create
        if self.train_size == None:
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

            self.ground_truth = dict((uid_to_int[key], value) for (key, value) in self.ground_truth.items())
            self.social_relations = dict((uid_to_int[key], value) for (key, value) in self.social_relations.items())
            if user_data:
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

    def not_in_ground_truth_message(self,uid):
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
            perfect_parameter_train = list(self.perfect_parameter.values())
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
                                                                self.poi_cats,self.undirected_category_tree,
                                                                self.final_rec_parameters['geo_div_method'])
            self.geo_div_propensity = self.pgeo_div_runner.compute_div_propensity()

            if self.final_rec_parameters['funnel'] == True:
                self.geo_div_propensity[self.geo_div_propensity>=0.5] = self.geo_div_propensity[self.geo_div_propensity>=0.5]**self.geo_div_propensity[self.geo_div_propensity>=0.5]
                self.geo_div_propensity[self.geo_div_propensity<0.5] = self.geo_div_propensity[self.geo_div_propensity<0.5]**(1+self.geo_div_propensity[self.geo_div_propensity<0.5])

        if self.final_rec_parameters['cat_div_method'] != None:
            self.pcat_div_runner = CatDivPropensity.getInstance(
                self.training_matrix,
                self.undirected_category_tree,
                self.final_rec_parameters['cat_div_method'],
                self.poi_cats)
            print("Computing categoric diversification propensity with",
                self.final_rec_parameters['cat_div_method'])
            self.cat_div_propensity=self.pcat_div_runner.compute_div_propensity()
            if self.final_rec_parameters['funnel'] == True:
                self.cat_div_propensity[self.cat_div_propensity>=0.5] = self.cat_div_propensity[self.cat_div_propensity>=0.5]**self.cat_div_propensity[self.cat_div_propensity>=0.5]
                self.cat_div_propensity[self.cat_div_propensity<0.5] = self.cat_div_propensity[self.cat_div_propensity<0.5]**(1+self.cat_div_propensity[self.cat_div_propensity<0.5])

        if self.final_rec_parameters['norm_method'] == 'default':
            if self.final_rec_parameters['cat_div_method'] == None:
                self.div_geo_cat_weight=1-self.geo_div_propensity
                self.div_weight=np.ones(len(self.div_geo_cat_weight))*self.final_rec_parameters['div_weight']
            elif self.final_rec_parameters['geo_div_method'] == None:
                self.div_geo_cat_weight=self.cat_div_propensity
                self.div_weight=np.ones(len(self.div_geo_cat_weight))*self.final_rec_parameters['div_weight']
            else:
                # self.div_geo_cat_weight=(self.cat_div_propensity)/(self.geo_div_propensity+self.cat_div_propensity)
                self.div_geo_cat_weight=self.cat_div_propensity*self.geo_div_propensity
                self.div_weight=np.ones(len(self.div_geo_cat_weight))*self.final_rec_parameters['div_weight']
                # self.div_weight = (self.geo_div_propensity+self.cat_div_propensity)/2
                # self.div_weight[(self.geo_div_propensity==0) & (self.cat_div_propensity==0)] = 0
                # groups = dict()
                # groups['geocat_preference'] = (self.div_geo_cat_weight >= 0.4) & (self.div_geo_cat_weight < 0.6)
                # groups['geo_preference'] = self.div_geo_cat_weight < 0.4
                # groups['cat_preference'] = self.div_geo_cat_weight >= 0.6

                # uid_group = dict()
                # for group, array in groups.items():
                #     uid_group.update(dict.fromkeys(np.nonzero(array)[0],group))
                # fgroups = open(self.data_directory+UTIL+f'groups_{self.get_final_rec_name()}.pickle','wb')
                # pickle.dump(uid_group,fgroups)
                # fgroups.close()
            
        elif self.final_rec_parameters['norm_method'] == 'quadrant':
            cat_div_propensity = self.cat_div_propensity
            geo_div_propensity = self.geo_div_propensity
            # cat_median = np.median(cat_div_propensity)
            # geo_median = np.median(geo_div_propensity)
            cat_median = 0.5
            geo_median = 0.5
            groups = dict()
            groups['geo_preference'] = (cat_div_propensity <= cat_median) & (geo_div_propensity > geo_median)
            groups['no_preference'] = ((cat_div_propensity <= cat_median) & (geo_div_propensity <= geo_median))
            groups['geocat_preference'] = ((cat_div_propensity >= cat_median) & (geo_div_propensity >= geo_median))
            groups['cat_preference'] = (cat_div_propensity > cat_median) & (geo_div_propensity <= geo_median)
            self.div_geo_cat_weight=np.zeros(self.training_matrix.shape[0])
            self.div_geo_cat_weight[groups['geo_preference']] = 0
            self.div_geo_cat_weight[groups['geocat_preference']] = 0.5
            self.div_geo_cat_weight[groups['cat_preference']] = 1
            self.div_weight=np.ones(len(self.div_geo_cat_weight))*self.final_rec_parameters['div_weight']
            self.div_weight[groups['no_preference']] = 0
            # self.div_geo_cat_weight[groups['no_preference']] = 0.25
            uid_group = dict()
            for group, array in groups.items():
                uid_group.update(dict.fromkeys(np.nonzero(array)[0],group))
            fgroups = open(self.data_directory+UTIL+f'groups_{self.get_final_rec_name()}.pickle','wb')
            pickle.dump(uid_group,fgroups)
            fgroups.close()

            
        if self.final_rec_parameters['bins'] != None:
            bins = np.append(np.arange(0,1,1/(self.final_rec_parameters['bins']-1)),1)
            centers = (bins[1:]+bins[:-1])/2
            self.div_geo_cat_weight = bins[np.digitize(self.div_geo_cat_weight, centers)]
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
            _cat_div_propensity=self.pcat_div_runner.compute_div_propensity()
            self.div_cat_weight=np.ones(len(self.div_geo_cat_weight))*_cat_div_propensity
        else:
            self.div_cat_weight=np.ones(len(self.div_geo_cat_weight))*self.final_rec_parameters['div_cat_weight']
        # print(self.div_cat_weight)
        # raise SystemExit

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
        self.binomial=Binomial.getInstance(self.training_matrix,self.poi_cats,self.cat_num,
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
        self.pm2 = Pm2(self.training_matrix,self.poi_cats,self.final_rec_parameters['div_weight'])
        # for uid in self.all_uids:
        #     self.run_pm2(uid)
        args=[(uid,) for uid in self.all_uids]
        results = run_parallel(self.run_pm2,args,self.CHKS)
        self.save_result(results,base=False)
        del self.pm2

    def perfectpersongeocat(self):
        if self.final_rec_parameters['train_size'] != None:
            user_checkin_data = dict()
            for uid, nuid in self.uid_to_int.items():
                user_checkin_data[nuid] = []
            for checkin in self.data_checkin_train:
                try:
                    user_checkin_data[self.uid_to_int[checkin['user_id']]].append({'poi_id':checkin['poi_id'],'date':checkin['date']})
                except:
                    pass

            tr_checkin_data=[]
            te_checkin_data=[]
            TRAIN_SIZE = self.final_rec_parameters['train_size']
            for i in tqdm(range(len(self.all_uids)),desc='train/test'):
                user_id=i
                checkin_list=user_checkin_data[user_id]
                checkin_list=sorted(checkin_list, key = lambda i: i['date']) 
                train_size=math.ceil(len(checkin_list)*TRAIN_SIZE)
                #test_size=math.floor(len(checkin_list)*TEST_SIZE)
                count=1
                te_pois=set()
                tr_pois=set()
                initial_te_size=len(te_checkin_data)
                final_te_size=len(te_checkin_data)
                for checkin in checkin_list:
                    if count<=train_size:
                        tr_pois.add(checkin['poi_id'])
                        tr_checkin_data.append({'user_id':user_id,'poi_id':checkin['poi_id'],'date':checkin['date']})
                    else:
                        te_pois.add(checkin['poi_id'])
                        te_checkin_data.append({'user_id':user_id,'poi_id':checkin['poi_id'],'date':checkin['date']})
                        final_te_size+=1
                    count+=1
                int_pois=te_pois&tr_pois
                rel_index=0
                for i in range(initial_te_size,final_te_size):
                    i+=rel_index
                    if te_checkin_data[i]['poi_id'] in int_pois:
                        te_checkin_data.pop(i)
                        rel_index-=1

            train_ground_truth = defaultdict(set)
            for checkin in te_checkin_data:
                train_ground_truth[checkin['user_id']].add(checkin['poi_id'])
            train_ground_truth = dict(train_ground_truth)

            train_training_matrix = np.zeros((self.user_num, self.poi_num))
            for checkin in tr_checkin_data:
                train_training_matrix[checkin['user_id'], checkin['poi_id']] += 1

            tmp_gt = self.ground_truth
            tmp_tm = self.training_matrix
            tmp_au = self.all_uids
            self.ground_truth = train_ground_truth
            self.training_matrix = train_training_matrix
            self.all_uids = list(train_ground_truth.keys())

        args=[(uid,) for uid in self.all_uids]
        results = run_parallel(self.run_perfectpersongeocat,args,self.CHKS)

        uids = [r[1] for r in results]
        div_geo_cat_weights = [r[2] for r in results]
        self.perfect_parameter = dict()
        for uid,div_geo_cat_weight in zip(uids,div_geo_cat_weights):
            self.perfect_parameter[uid] = div_geo_cat_weight
        print(list(self.perfect_parameter.keys()))
        # print(self.perfect_parameter)
        results = [r[0] for r in results]

        fout = open(self.data_directory+UTIL+f'parameter_{self.get_final_rec_name()}.pickle',"wb")
        pickle.dump(self.perfect_parameter,fout)
        self.save_result(results,base=False)

        if self.final_rec_parameters['train_size'] != None:
            self.ground_truth = tmp_gt
            self.training_matrix = tmp_tm
            self.all_uids = tmp_au

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
        self.not_in_ground_truth_message(uid)
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
        self.not_in_ground_truth_message(uid)
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
                                                             'local_max',
                                                             gcobjfunc.OBJECTIVE_FUNCTIONS.get('cat_weight'),
                                                             self.final_rec_parameters['div_cat_weight'])

                precision_val=metrics.recallk(actual, predicted[:self.final_rec_parameters['k']])
                if precision_val > max_predicted_val:
                    max_predicted_val = precision_val
                    max_predicted = predicted
                    max_overall_scores = overall_scores
                    max_div_geo_cat_weight = div_geo_cat_weight
                    # print("%d uid with geocatweight %f" % (uid,div_geo_cat_weight))
                    # print(self.perfect_parameter)
            predicted, overall_scores = max_predicted, max_overall_scores
            return json.dumps({'user_id': uid, 'predicted': list(map(int, predicted)), 'score': list(map(float, overall_scores))})+"\n", uid, max_div_geo_cat_weight
        self.not_in_ground_truth_message(uid)
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
        self.not_in_ground_truth_message(uid)
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
        self.not_in_ground_truth_message(uid)
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
        self.not_in_ground_truth_message(uid)
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
                                                         gcobjfunc.OBJECTIVE_FUNCTIONS.get(self.final_rec_parameters['obj_func']),
                                                         self.final_rec_parameters['div_cat_weight'])

            return json.dumps({'user_id': uid, 'predicted': list(map(int, predicted)), 'score': list(map(float, overall_scores))})+"\n"
        self.not_in_ground_truth_message(uid)
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
            # if math.isnan(div_geo_cat_weight):
            #     print(f"User {uid} with nan value")
            div_weight=self.div_weight[uid]
            div_cat_weight=self.div_cat_weight[uid]
            predicted, overall_scores = gcobjfunc.geocat(uid, self.training_matrix, predicted, overall_scores,
                                                         self.poi_cats, self.poi_neighbors, self.final_rec_list_size, self.undirected_category_tree,
                                                         div_geo_cat_weight,div_weight,
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
        self.not_in_ground_truth_message(uid)
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
        self.not_in_ground_truth_message(uid)
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
        self.not_in_ground_truth_message(uid)
        return ""

    def load_base_predicted(self):
        if self.k_fold == None:
            folds=  [None]
        else:
            folds = range(self.k_fold)

        for self.fold in folds:
            
            result_file = open(self.data_directory+"result/reclist/"+self.get_base_rec_file_name(), 'r')
            for i,line in enumerate(result_file):
                obj=json.loads(line)
                self.user_base_predicted_lid[obj['user_id']]=obj['predicted']
                self.user_base_predicted_score[obj['user_id']]=obj['score']
            self.recs_user_base_predicted_lid[self.get_base_rec_name()] = self.user_base_predicted_lid
            self.recs_user_base_predicted_score[self.get_base_rec_name()] = self.user_base_predicted_score
        


    def load_final_predicted(self):
        if self.k_fold == None:
            folds=  [None]
        else:
            folds = range(self.k_fold)

        for self.fold in folds:
            result_file = open(self.data_directory+"result/reclist/"+self.get_final_rec_file_name(), 'r')
            self.user_final_predicted_lid = dict()
            self.user_final_predicted_score = dict()
            for i,line in enumerate(result_file):
                obj=json.loads(line)
                self.user_final_predicted_lid[obj['user_id']]=obj['predicted']
                self.user_final_predicted_score[obj['user_id']]=obj['score']
            self.recs_user_final_predicted_lid[self.get_final_rec_name()] = self.user_final_predicted_lid
            self.recs_user_final_predicted_score[self.get_final_rec_name()] = self.user_final_predicted_score

    def message_recommender(self,base):
        if base:
            print(f"{self.base_rec} base recommender")
        else:
            print(f"{self.final_rec} final recommender")
        self.print_parameters(base=base)
        print(f"Base rec list size = {self.base_rec_list_size}")
        print(f"Final rec list size = {self.final_rec_list_size}")


    def create_train_validation(self):
        user_checkin_data = defaultdict(list)
        # for uid, nuid in self.uid_to_int.items():
        #     user_checkin_data[nuid] = []

        for checkin in self.data_checkin_train:
            user_checkin_data[checkin['user_id']].append({'poi_id':checkin['poi_id'],'date':checkin['date']})
        all_uids = list(user_checkin_data.keys())
        user_num = len(all_uids)
        tr_checkin_data=[]
        te_checkin_data=[]
        TRAIN_SIZE = self.train_size
        for i in tqdm(range(len(all_uids)),desc='train/test'):
            user_id=i
            checkin_list=user_checkin_data[user_id]
            checkin_list=sorted(checkin_list, key = lambda i: i['date']) 
            train_size=math.ceil(len(checkin_list)*TRAIN_SIZE)
            #test_size=math.floor(len(checkin_list)*TEST_SIZE)
            count=1
            te_pois=set()
            tr_pois=set()
            initial_te_size=len(te_checkin_data)
            final_te_size=len(te_checkin_data)
            for checkin in checkin_list:
                if count<=train_size:
                    tr_pois.add(checkin['poi_id'])
                    tr_checkin_data.append({'user_id':user_id,'poi_id':checkin['poi_id'],'date':checkin['date']})
                else:
                    te_pois.add(checkin['poi_id'])
                    te_checkin_data.append({'user_id':user_id,'poi_id':checkin['poi_id'],'date':checkin['date']})
                    final_te_size+=1
                count+=1
            int_pois=te_pois&tr_pois
            rel_index=0
            for i in range(initial_te_size,final_te_size):
                i+=rel_index
                if te_checkin_data[i]['poi_id'] in int_pois:
                    te_checkin_data.pop(i)
                    rel_index-=1

        ground_truth = defaultdict(set)
        for checkin in te_checkin_data:
            ground_truth[checkin['user_id']].add(checkin['poi_id'])
        ground_truth = dict(ground_truth)

        training_matrix = np.zeros((user_num, self.poi_num),dtype=np.float32)
        for checkin in tr_checkin_data:
            training_matrix[checkin['user_id'], checkin['poi_id']] += 1
        ftecheckin=open(self.data_directory+UTIL+f'test_val_{self.get_train_validation_name()}.pickle','wb')
        pickle.dump(ground_truth,ftecheckin)
        ftecheckin.close()
        ftrcheckin=open(self.data_directory+UTIL+f'train_val_{self.get_train_validation_name()}.pickle','wb')
        pickle.dump(training_matrix,ftrcheckin)
        ftrcheckin.close()

    def get_train_validation_name(self):
        return f'{self.train_size}_{self.city}'

    def create_fold_train_test(self):
        user_checkin_data = defaultdict(list)

        for checkin in self.data_checkin_train:
            user_checkin_data[checkin['user_id']].append({'poi_id':checkin['poi_id'],'date':checkin['date']})

        tr_checkin_data=[]
        te_checkin_data=[]
        TRAIN_SIZE = 1-1/self.k_fold
        TEST_SIZE = 1-TRAIN_SIZE
        for i in tqdm(range(len(self.all_uids)),desc='train/test'):
            user_id=i
            checkin_list=user_checkin_data[user_id]
            checkin_list=sorted(checkin_list, key = lambda i: i['date']) 
            train_size=math.ceil(len(checkin_list)*TRAIN_SIZE)
            test_size = len(checkin_list)-train_size
            test_init = test_size*self.fold
            test_end = test_size*(self.fold+1)
            #test_size=math.floor(len(checkin_list)*TEST_SIZE)
            count=1
            te_pois=set()
            tr_pois=set()
            initial_te_size=len(te_checkin_data)
            final_te_size=len(te_checkin_data)
            for checkin in checkin_list:
                if test_init <= count and count < test_end:
                    te_pois.add(checkin['poi_id'])
                    te_checkin_data.append({'user_id':user_id,'poi_id':checkin['poi_id'],'date':checkin['date']})
                    final_te_size+=1
                else:
                    tr_pois.add(checkin['poi_id'])
                    tr_checkin_data.append({'user_id':user_id,'poi_id':checkin['poi_id'],'date':checkin['date']})
                count+=1
            int_pois=te_pois&tr_pois
            rel_index=0
            for i in range(initial_te_size,final_te_size):
                i+=rel_index
                if te_checkin_data[i]['poi_id'] in int_pois:
                    te_checkin_data.pop(i)
                    rel_index-=1

        ground_truth = defaultdict(set)
        for checkin in te_checkin_data:
            ground_truth[checkin['user_id']].add(checkin['poi_id'])
        ground_truth = dict(ground_truth)

        training_matrix = np.zeros((self.user_num, self.poi_num))
        for checkin in tr_checkin_data:
            training_matrix[checkin['user_id'], checkin['poi_id']] += 1
        ftecheckin=open(self.data_directory+UTIL+f'test_{self.get_fold_name()}.pickle','wb')
        pickle.dump(ground_truth,ftecheckin)
        ftecheckin.close()
        ftrcheckin=open(self.data_directory+UTIL+f'train_{self.get_fold_name()}.pickle','wb')
        pickle.dump(training_matrix,ftrcheckin)
        ftrcheckin.close()

    def get_fold_name(self):
        return f'{self.k_fold}_{self.fold}_{self.city}'
        
    def create_k_fold(self):
        for fold in range(self.k_fold):
            self.fold = fold
            self.create_fold_train_test()
        pass

    def load_fold(self):
        self.training_matrix = pickle.load(open(self.data_directory+UTIL+f'train_{self.get_fold_name()}.pickle','rb'))
        self.ground_truth = pickle.load(open(self.data_directory+UTIL+f'test_{self.get_fold_name()}.pickle','rb'))
    
    def run_base_recommender(self,check_already_exists=False):

        if check_already_exists == True and os.path.exists(self.data_directory+RECLIST+self.get_base_rec_file_name()):
            print("recommender not going to be ran, already generated %s" % (self.get_base_rec_name()))
            return
        base_recommender=self.BASE_RECOMMENDERS[self.base_rec]

        self.message_recommender(base=True)
        start_time = time.time()
        if self.k_fold != None:
            for self.fold in range(self.k_fold):
                self.load_fold()
                base_recommender()
        else:
            base_recommender()

        final_time = time.time()-start_time
        fout = open(self.data_directory+UTIL+f'run_time_{self.get_base_rec_name()}.txt',"w")
        fout.write(str(final_time))
        fout.close()

    def run_final_recommender(self,check_already_exists=False):
        if check_already_exists == True and os.path.exists(self.data_directory+RECLIST+self.get_final_rec_file_name()):
            print("recommender not going to be ran, already generated %s" % (self.get_final_rec_name()))
            return
        final_recommender=self.FINAL_RECOMMENDERS[self.final_rec]
        if len(self.user_base_predicted_lid)>0:
            self.message_recommender(base=False)
            start_time = time.time()

            if self.k_fold != None:
                for self.fold in range(self.k_fold):
                    self.load_fold()
                    final_recommender()
            else:
                final_recommender()
            final_time = time.time()-start_time
            fout = open(self.data_directory+UTIL+f'run_time_{self.get_final_rec_name()}.txt',"w")
            fout.write(str(final_time))
            fout.close()
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
        f1_val=metrics.f1k(precision_val,rec_val)
        # map_val = metrics.mapk(actual,predicted_at_k,k)
        # ndcg_val = metrics.ndcgk(actual,predicted_at_k,k)
        # ildg_val = metrics.ildgk(predicted_at_k,self.poi_coos)
        # else:
        #     epc_val=self.epc_val
        # if uid == max(self.all_uids):
        #     del self.epc_val
        # this epc is maded like vargas, recsys'11
        #epc_val=metrics.epck(predicted_at_k,actual,uid,self.training_matrix)
        
        d={'user_id':uid,'precision':precision_val,'recall':rec_val,'pr':pr_val,'ild':ild_val,'gc':gc_val,'epc':epc_val,'f1': f1_val,
           # 'map':map_val,'ndcg':ndcg_val,
           # 'ildg': ildg_val
        }

        return json.dumps(d)+'\n'

    def get_file_name_metrics(self,base,k):
        if base:
            return self.data_directory+"result/metrics/"+self.get_base_rec_name()+f"_{str(k)}{R_FORMAT}"
        else:
            return self.data_directory+"result/metrics/"+self.get_final_rec_name()+f"_{str(k)}{R_FORMAT}"

    def eval_rec_metrics(self,*,base=False,METRICS_KS = experiment_constants.METRICS_K,eval_group=False):

        if eval_group:
            tmp_final_rec = self.final_rec
            tmp_final_rec_parameters = self.final_rec_parameters
            self.final_rec = 'persongeocat'
            self.final_rec_parameters = {'cat_div_method': 'inv_num_cat', 'geo_div_method': 'walk', 'norm_method': 'quadrant','bins': None,'funnel':None}
            uid_group = self.get_groups()
            groups_count = Counter(uid_group.values())
            unique_groups = list(groups_count.keys())
            self.final_rec = tmp_final_rec
            self.final_rec_parameters = tmp_final_rec_parameters
            groups_epc = dict()
        
        if self.k_fold == None:
            folds=  [None]
        else:
            folds = range(self.k_fold)
        for self.fold in folds:
            if base:
                predictions = self.recs_user_base_predicted_lid[self.get_base_rec_name()]
                self.user_base_predicted_lid = predictions
            else:
                predictions = self.recs_user_final_predicted_lid[self.get_final_rec_name()]
                self.user_final_predicted_lid = predictions
            if self.k_fold == None:
                all_uids = self.all_uids
            else:
                all_uids = predictions.keys()
            for i,k in enumerate(METRICS_KS):
                if (k <= self.final_rec_list_size and not base) or (k <= self.base_rec_list_size and base):
                    print(f"running metrics at @{k}")
                    self.epc_val = metrics.old_global_epck(self.training_matrix,self.ground_truth,predictions,predictions.keys(),k)
                    if eval_group:
                        groups_epc[k] = dict()
                        for group in unique_groups:
                            uids = [uid for uid, gid in uid_group.items() if gid == group]
                            groups_epc[k][group] = metrics.old_global_epck(self.training_matrix,self.ground_truth,predictions,all_uids,k)

                    # if base:
                    result_out = open(self.get_file_name_metrics(base,k), 'w')
                    # else:
                    #     result_out = open(self.data_directory+"result/metrics/"+self.get_final_rec_name()+f"_{str(k)}{R_FORMAT}", 'w')

                    self.message_recommender(base=base)

                    args=[(uid,base,k) for uid in all_uids]
                    results = run_parallel(self.eval,args,self.CHKSL)
                    print(pd.DataFrame([json.loads(result) for result in results]).mean().T)
                    # print(pd.DataFrame([json.loads(result) for result in results]).std().T)
                    for json_string_result in results:
                        result_out.write(json_string_result)
                    result_out.close()
                else:
                    print(f"Trying to evaluate list with @{k}, greather than final rec list size")

        if eval_group:
            fout = open(self.data_directory+UTIL+f'groups_epc_{self.get_final_rec_name()}.pickle',"wb")
            pickle.dump(groups_epc,fout)
            fout.close()

    def load_metrics(self,*,base=False,name_type=NameType.PRETTY,METRICS_KS=experiment_constants.METRICS_K,epc_group=False):
        if base:
            rec_using=self.base_rec
            if name_type == NameType.PRETTY:
                rec_short_name=self.get_base_rec_pretty_name()
            elif name_type == NameType.SHORT:
                rec_short_name=self.get_base_rec_short_name()
            elif name_type == NameType.FULL:
                rec_short_name=self.get_base_rec_name()
        else:
            rec_using=self.final_rec
            if name_type == NameType.BASE_PRETTY:
                rec_short_name=self.get_final_rec_pretty_name()+'('+self.get_base_rec_pretty_name()+')'
            elif name_type == NameType.PRETTY:
                rec_short_name=self.get_final_rec_pretty_name()
            elif name_type == NameType.SHORT:
                rec_short_name=self.get_final_rec_short_name()
            elif name_type == NameType.CITY_BASE_PRETTY:
                rec_short_name=f"({CITIES_PRETTY[self.city]})"+self.get_final_rec_pretty_name()+'('+self.get_base_rec_pretty_name()+')'
            elif name_type == NameType.FULL:
                rec_short_name=self.get_final_rec_name()

        print("Loading %s..." % (rec_short_name))

        if epc_group:
            fin = open(self.data_directory+UTIL+f'groups_epc_{self.get_final_rec_name()}.pickle',"rb")
            self.groups_epc[rec_short_name] = pickle.load(fin)
            fin.close()
        self.metrics[rec_short_name]={}
        self.metrics_cities[rec_short_name] = self.city
        for i,k in enumerate(METRICS_KS):
            try:
                result_file = open(self.get_file_name_metrics(base,k), 'r')

                self.metrics[rec_short_name][k]=[]
                for i,line in enumerate(result_file):
                    obj=json.loads(line)
                    self.metrics[rec_short_name][k].append(obj)
            except Exception as e:
                print(e)
        if self.metrics[rec_short_name] == {}:
            del self.metrics[rec_short_name]


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
            fig = plt.figure(figsize=(8, 6))
            ax=fig.add_subplot(111)
            # ax.grid(alpha=MPL_ALPHA,axis='y')
            num_recs_plot = len(self.metrics)-1
            barWidth= 1-num_recs_plot/(1+num_recs_plot)
            N=len(self.metrics_name)
            indexes=np.arange(N)
            i=0
            styles = gen_bar_cycle(num_recs_plot)()
            y_to_annotate = 90
            neg_y_to_annotate = 2
            put_annotation = []
            rects = dict()
            for rec_using,rec_metrics in metrics_mean.items():
                print(f"{rec_using} at @{k}")
                print(rec_metrics)
                rel_diffs = -100+100*np.array(list(rec_metrics.values()))/reference_vals
                rects[rec_using] = ax.bar(indexes+i*barWidth,rel_diffs,barWidth,label=rec_using,**next(styles))[0]
                special_cases = rel_diffs > 100
                for idx, value in zip(indexes[special_cases],rel_diffs[special_cases]):
                    ax.annotate(f'{int(value)}%',xy=(idx+i*barWidth-barWidth,y_to_annotate),zorder=25,color='k',fontsize=18,weight='bold')
                    y_to_annotate -= 10

                special_cases = rel_diffs < -25

                for idx, value in zip(indexes[special_cases],rel_diffs[special_cases]):
                    ax.annotate(f'{int(value)}%',xy=(idx+i*barWidth-barWidth,neg_y_to_annotate),zorder=25,color='k',fontsize=18,weight='bold')
                    neg_y_to_annotate += 10
                i+=1
            rects = list(rects.values())
                #ax.bar(indexes[j]+i*barWidth,np.mean(list(rec_metrics.values())),barWidth,label=rec_using,color=palette(i))
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
            ax.set_xticks(np.arange(N)+barWidth*(((num_recs_plot)/2)-1)+barWidth/2)
            # ax.legend((p1[0], p2[0]), self.metrics_name)
            ax.legend(rects,tuple(metrics_mean.keys()),bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                      mode="expand", borderaxespad=0, ncol=3,fontsize=19,handletextpad=-0.6,
                      handler_map={rect: HandlerSquare() for rect in rects})
            # ax.legend(tuple(map(lambda name: METRICS_PRETTY[name],self.metrics.keys())))
            ax.set_xticklabels(map(lambda x: f'{x}@{k}',list(map(lambda name: METRICS_PRETTY[name],self.metrics_name))),fontsize=17)
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(19) 
            if LANG == 'pt':
                ax.set_ylabel(f"Diferença relativa ao {RECS_PRETTY[self.base_rec]}",fontsize=23)
            else:
                ax.set_ylabel(f"Diferença relativa ao {RECS_PRETTY[self.base_rec]}",fontsize=23)
            ax.set_ylim(-25,100)
            ax.set_xlim(-barWidth,len(self.metrics_name)-1+(num_recs_plot-1)*barWidth+barWidth)

            for i, m in enumerate(self.metrics_name):
                ax.axvline(i+num_recs_plot*barWidth,color='k',linewidth=1,alpha=0.5)
            # ax.set_title(f"at @{k}, {self.city}")
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())
            fig.show()
            plt.show()
            timestamp = datetime.timestamp(datetime.now())
            fig.savefig(self.data_directory+f"result/img/{prefix_name}_{self.base_rec}_{self.city}_{str(k)}.png",bbox_inches="tight")
            fig.savefig(self.data_directory+f"result/img/{prefix_name}_{self.base_rec}_{self.city}_{str(k)}.eps",bbox_inches="tight")

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
                self.invalid_uids.append(i)
            if train_size == 0:
                print(f"user {i} with empty training data!!!!! Really bad error")
                has_some_error = True
            if has_some_error:
                has_some_error_global = True
                print("Training size is %d, test size is %d" %\
                      (train_size,test_size))
        for uid in self.invalid_uids:
            self.all_uids.remove(uid)

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

        fig=plt.figure()
        ax = fig.subplots(1,1)

        labels, counts = np.unique(vals, return_counts=True)
        ax.bar(labels, counts, align='center',width=0.25,color='k')
        ax.set_xticks(labels)
        for label, count in zip(labels,counts):
            ax.text(label, count, count,ha='center')
        # plt.plot(vals)
        # plt.hist(vals)
        fig.savefig(self.data_directory+IMG+f'perf_param_{self.get_final_rec_name()}.png',bbox_inches="tight")
        # plt.savefig(self.data_directory+IMG+f'perfect_{self.city}.png')


    def generate_general_user_data(self):
        preprocess_methods = [# [None,'walk']
                              # ,['poi_ild',None]
        ]
        for cat_div_method in [None]+CatDivPropensity.METHODS:
            for geo_div_method in [None]+GeoDivPropensity.METHODS:
                if cat_div_method != None or geo_div_method != None:
                    preprocess_methods.append([cat_div_method,geo_div_method])

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
        # self.final_rec = 'perfectpgeocat'
        fin = open(self.data_directory+UTIL+f'parameter_{self.get_final_rec_name()}.pickle',"rb")
        self.perfect_parameter = pickle.load(fin)
        # vals = np.array([])
        # for uid, val in self.perfect_parameter.items():
        #     vals = np.append(vals,val)
        # self.perfect_parameter = vals
        # self.perfect_parameter = pd.Series(self.perfect_parameter)
        self.final_rec = final_rec
    
    def print_correlation_perfectparam(self):
        # self.base_rec = 'usg'

        self.load_perfect()
        self.perfect_parameter = pd.Series(self.perfect_parameter.values())

        df_poly=self.generate_general_user_data()

        correlations = df_poly.corrwith(list(self.perfect_parameter.values()))
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
        perfect_parameter_test = list(self.perfect_parameter.values())
        perfect_parameter_train = np.array([])
        df = pd.DataFrame()
        for city in experiment_constants.CITIES:
            if city != old_city:
                self.city = city
                self.load_base(user_data=True)
                self.load_perfect()
                perfect_parameter_train = np.append(perfect_parameter_train,list(self.perfect_parameter.values()))
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

    def print_latex_metrics_table(self,prefix_name='',references=[],heuristic=False,cities=None):
        num_metrics = len(self.metrics_name)
        if heuristic:
            num_metrics += 1
        result_str = r"\begin{table}[]" + "\n"
        result_str += r"\begin{tabular}{" +'l|'+'l'*(num_metrics) + "}\n"
        # result_str += "\begin{tabular}{" + 'l'*(num_metrics+1) + "}\n"

        if cities == None:
            cities = [self.city]
        final_rec_list_size = self.final_rec_list_size

        for city in cities:
            self.city = city
            self.final_rec_list_size = final_rec_list_size
            if heuristic:
                self.load_metrics(base=True,name_type=NameType.PRETTY)

            run_times = dict()

            if not references:
                references = [list(self.metrics.keys())[0]]

            result_str += "\t&"+'\multicolumn{%d}{c}{%s}\\\\\n' % (num_metrics,CITIES_PRETTY[city])

            if heuristic:
                self.final_rec_parameters = {'heuristic': 'local_max'}
                self.load_metrics(base=False,name_type=NameType.PRETTY)

            for i,k in enumerate(experiment_constants.METRICS_K):


                if heuristic:
                    self.final_rec_list_size = k
                    for h in ['tabu_search', 'particle_swarm']:
                        self.final_rec_parameters = {'heuristic': h}
                        self.load_metrics(base=False,name_type=NameType.PRETTY,METRICS_KS=[k])
                        run_times[self.get_final_rec_pretty_name()]=int(float(open(self.data_directory+UTIL+f'run_time_{self.get_final_rec_name()}.txt',"r").read()))
                    self.final_rec_parameters = {'heuristic': 'local_max'}
                    run_times[self.get_final_rec_pretty_name()]=int(float(open(self.data_directory+UTIL+f'run_time_{self.get_final_rec_name()}.txt',"r").read()))
                    max_arg_run_time = max(run_times, key=run_times.get)

                result_str += "\\hline \\rowcolor{Gray} \\textbf{Algorithm} & "+'& '.join(map(lambda x: "\\textbf{"+METRICS_PRETTY[x]+f"@{k}}}" ,self.metrics_name))

                if heuristic:
                    result_str += '& \\textbf{Time}'
                result_str += "\\\\\n"
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
                                metrics_gain[rec_using][metric_name] = bullet_str
                            else:
                                if metrics_mean[rec_using][metric_name] < metrics_mean[reference_name][metric_name]:
                                    metrics_gain[rec_using][metric_name] = triangle_down_str
                                elif metrics_mean[rec_using][metric_name] > metrics_mean[reference_name][metric_name]:
                                    metrics_gain[rec_using][metric_name] = triangle_up_str
                                else:
                                    metrics_gain[rec_using][metric_name] = bullet_str 
                    else:
                        reference_name = rec_using
                        base_metrics = metrics
                        metrics_gain[rec_using] = dict()
                        for metric_name in self.metrics_name:
                            metrics_gain[rec_using][metric_name] = ''

                metrics_max = {mn:(None,0) for mn in self.metrics_name}
                for rec_using,rec_metrics in metrics_mean.items():
                    for metric_name, value in rec_metrics.items():
                        if metrics_max[metric_name][1] < value:
                            metrics_max[metric_name] = (rec_using,value)
                is_metric_the_max = defaultdict(lambda: defaultdict(bool))
                for metric_name,(rec_using,value) in metrics_max.items():
                    is_metric_the_max[rec_using][metric_name] = True

                for rec_using,rec_metrics in metrics_mean.items():
                    gain = metrics_gain[rec_using]
                    result_str += rec_using + ' &' + '& '.join(map(lambda x: "\\textbf{%.4f}%s" %(x[0],x[1]) if is_metric_the_max[rec_using][x[2]] else "%.4f%s"%(x[0],x[1])  ,zip(rec_metrics.values(),gain.values(),rec_metrics.keys())))
                    if heuristic:
                        if rec_using not in references:
                            if rec_using != max_arg_run_time:
                                result_str += f'& {sec_to_hm(run_times[rec_using])}'
                            else:
                                result_str += f'& \\textbf{{{sec_to_hm(run_times[rec_using])}}}'
                        else:
                            result_str += f'& '
                    result_str += "\\\\\n"
            if city != cities[-1]:
                result_str += '\\hline'


        result_str += "\\end{tabular}\n"
        result_str += "\\end{table}\n"
        result_str = LATEX_HEADER + result_str
        result_str += LATEX_FOOT
        fout = open(self.data_directory+UTIL+'_'.join(([prefix_name] if len(prefix_name)>0 else [])+cities)+'.tex', 'w')
        fout.write(result_str)
        fout.close()

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
            ax.set_xticklabels(map(lambda x: f'{x}@{k}',list(map(lambda name: METRICS_PRETTY[name],self.metrics_name))))
            # ax.set_title(f"at @{k}, {self.city}")
            ax.set_ylabel("Mean Value")
            ax.set_ylim(0,1)
            ax.set_xlim(-barWidth,len(self.metrics_name)-1+(len(self.metrics)-1)*barWidth+barWidth)
            fig.show()
            plt.show()
            timestamp = datetime.timestamp(datetime.now())
            fig.savefig(self.data_directory+f"result/img/{prefix_name}_{self.base_rec}_{self.city}_{str(k)}.png",bbox_inches="tight")

    def plot_maut(self,prefix_name='maut',ncol=4,print_times=False,print_text=True):
        # palette = plt.get_cmap(CMAP_NAME)
        fig = plt.figure()
        ax=fig.add_subplot(111)
        num_recs = len(self.metrics)
        N=len(experiment_constants.METRICS_K)
        barWidth=1-len(self.metrics)/(1+len(self.metrics))
        indexes=np.arange(N)
        rects = dict()
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
            # print(get_my_color_scheme(len(self.metrics)))
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
                rects[rec_using] = ax.bar(count+i*barWidth,val,barWidth,**next(styles),label=rec_using)[0]
                    # ax.bar(count+i*barWidth,val,barWidth,**GEOCAT_BAR_STYLE,label=rec_using)
                if print_text:
                    ax.text(count+i*barWidth-barWidth/2+barWidth*0.1,val+0.025,"%.2f"%(val),rotation=90,zorder=33)
                i+=1
            #ax.set_xticks(np.arange(N+1)+barWidth*(np.floor((len(self.metrics))/2)-1)+barWidth/2)
        # ax.set_xticks(np.arange(N+1)+barWidth*len(metrics_utility_score)/2+barWidth/2)


        rects = list(rects.values())
        ax.set_xticks(np.arange(N)+barWidth*(((len(self.metrics))/2)-1)+barWidth/2)
        # ax.legend((p1[0], p2[0]), self.metrics_name)
        ax.legend(rects,tuple(self.metrics.keys()),bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                  mode="expand", borderaxespad=0, ncol=ncol,fontsize=15,handletextpad=-0.6,
                  handler_map={rect: HandlerSquare() for rect in rects})
        # ax.legend(tuple(map(lambda name: METRICS_PRETTY[name],self.metrics.keys())))
        ax.set_xticklabels(['@5','@10','@20'],fontsize=16)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(16)
        # ax.set_title(f"at @{k}, {self.city}")
        ax.set_ylabel("MAUT",fontsize=18)
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

        for i in range(N):
            ax.axvline(i+num_recs*barWidth,color='k',linewidth=1,alpha=0.5)
        fig.show()
        plt.show()

        timestamp = datetime.timestamp(datetime.now())
        fig.savefig(self.data_directory+f"result/img/{prefix_name}_{self.base_rec}_{self.city}.png",bbox_inches="tight")
        fig.savefig(self.data_directory+f"result/img/{prefix_name}_{self.base_rec}_{self.city}.eps",bbox_inches="tight")

    def print_ild_gc_correlation(self,metrics=['ild','gc']):
        rec = list(self.metrics.keys())[-1]
        metrics_ks = list(self.metrics.values())[-1]

        for k, metrics_k in metrics_ks.items():
            print("%s@%d correlation"%(rec,k))
            df_p_metrics = pd.DataFrame(metrics_k)
            df_p_metrics = df_p_metrics.set_index('user_id')
            print(df_p_metrics[metrics].corr().iloc[0,1])

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
    
    def plot_geo_cat_methods(self,diff_to_optimal=False,default_label_axis=True,default_legend=True,plot_ord=False,plot_cdf=False):

        if diff_to_optimal:
            final_rec = self.final_rec
            self.final_rec = 'perfectpgeocat'
            self.load_perfect()
            perfect_parameter = self.perfect_parameter
            self.final_rec = final_rec
        
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
        # print("user with most pois")
        # print(np.nonzero(self.training_matrix[np.argmax(np.count_nonzero(self.training_matrix,axis=1)),:])[0])
        # print([self.poi_coos[lid] for lid in np.nonzero(self.training_matrix[np.argmax(np.count_nonzero(self.training_matrix,axis=1)),:])[0]])

        # self.final_rec_parameters = {'cat_div_method': 'inv_num_cat', 'geo_div_method': 'walk', 'norm_method': 'quadrant','bins': None,'funnel':None}
        # uid_group = self.get_groups()
        # groups_count = Counter(uid_group.values())
        # unique_groups = list(groups_count.keys())
        # unique_groups = [x for y, x in sorted(zip(iter(map(GROUP_ID.get,unique_groups)), unique_groups))]
        for geo_div_method in geo_div_methods:
            for cat_div_method in cat_div_methods:
                geo_div_propensity = geo_div_propensities[geo_div_method]
                cat_div_propensity = cat_div_propensities[cat_div_method]
                fig = plt.figure(figsize=(8,8))

                plt.rcParams.update({'font.size': 18})
                ax = fig.add_subplot(111)
                # ax.set_ylim(0,1)
                # cat_median = np.median(cat_div_propensity)
                # geo_median = np.median(geo_div_propensity)

                cat_median = 0.5
                geo_median = 0.5
                groups = dict()
                
                groups['cat_preference'] = (cat_div_propensity > cat_median) & (geo_div_propensity <= geo_median)
                groups['geo_preference'] = (cat_div_propensity <= cat_median) & (geo_div_propensity > geo_median)

                groups['geocat_preference'] = ((cat_div_propensity >= cat_median) & (geo_div_propensity >= geo_median))
                groups['no_preference'] = ((cat_div_propensity <= cat_median) & (geo_div_propensity <= geo_median))

                assert(np.max(groups['no_preference'] & groups['cat_preference'] & groups['geo_preference'] & groups['geocat_preference']) == 0)
                unique_groups = GROUP_ID.keys()
                colors = get_my_color_scheme(len(groups),ord_by_brightness=True,inverse_order=False)
                for group, color in zip(unique_groups,colors):
                    mask = groups[group]
                    # print(mask)
                    # print(len(cat_div_propensity[mask]))
                    # print(len(geo_div_propensity[mask]))
                    ax.scatter(cat_div_propensity[mask],geo_div_propensity[mask],color=color)
                # ax.plot([cat_median]*2,[0,1],color='k')
                # ax.plot([0,1],[geo_median]*2,color='k')
                # ax.plot([cat_median]*2,ax.get_ylim(),color='k')
                # ax.plot(ax.get_ylim(),[geo_median]*2,color='k')
                # ax.set_xlim([min(cat_div_propensity),max(cat_div_propensity)])
                # ax.set_ylim([min(geo_div_propensity),max(geo_div_propensity)])
                if not default_label_axis:
                    ax.set_xlabel('Categorical method ('+CatDivPropensity.CAT_DIV_PROPENSITY_METHODS_PRETTY_NAME[cat_div_method]+')')
                    ax.set_ylabel('Geographic method ('+GeoDivPropensity.GEO_DIV_PROPENSITY_METHODS_PRETTY_NAME[geo_div_method]+')')
                else:
                    if LANG == 'pt':
                        ax.set_xlabel(r'Parâmetro de div-cat ( $\theta_{cat}$ )')
                        ax.set_ylabel(r'Parâmetro de div-geo ( $\theta_{geo}$ )')
                    else:
                        ax.set_xlabel(r'Parameter of Cat-Diversification ( $\theta_{cat}$ )')
                        ax.set_ylabel(r'Parameter of Geo-Diversification ( $\theta_{geo}$ )')
                    
                # ax.set_xlim(min(np.min(cat_div_propensity),0),max(np.max(cat_div_propensity),1))
                # ax.set_ylim(min(np.min(geo_div_propensity),0),max(np.max(geo_div_propensity),1))

                # ax.set_title("Correlation: %f"%(scipy.stats.pearsonr(cat_div_propensity,geo_div_propensity)[0]))
                ax.annotate("Pearson = %.2f"%(scipy.stats.pearsonr(cat_div_propensity,geo_div_propensity)[0]),
                                (0,0))
                if not default_legend:
                    ax.legend((
                        f"Categorical preference ({np.count_nonzero(groups['cat_preference'])} users)",
                        f"Geographical preference ({np.count_nonzero(groups['geo_preference'])} users)",
                        f"Equal preference ({np.count_nonzero(groups['geocat_preference'])} users)",
                        f"No preference ({np.count_nonzero(groups['no_preference'])} users)",
                    ))
                else:
                    if LANG == 'pt':
                        ax.legend([f"G{GROUP_ID[group]} ({np.count_nonzero(groups[group])} usuários)" for group in unique_groups]
                                ,handletextpad=-0.6,scatteryoffsets=[0.5],
                                bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                                mode="expand", borderaxespad=0, ncol=2
                        )
                    else:
                        ax.legend([f"Group {GROUP_ID[group]} ({np.count_nonzero(groups[group])} users)" for group in unique_groups]
                                ,handletextpad=-0.6,scatteryoffsets=[0.5],
                                bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                                mode="expand", borderaxespad=0, ncol=2
                        )
                # ax.plot([cat_median]*2,[min(geo_div_propensity),max(geo_div_propensity)],color='k')
                # ax.plot([min(cat_div_propensity),max(cat_div_propensity)],[geo_median]*2,color='k')
                ax.axvline(0.5,color='k',linewidth=1)
                ax.axhline(0.5,color='k',linewidth=1)
                ax.set_xticks(np.linspace(0,1,6))
                ax.set_yticks(np.linspace(0,1,6))
                fig.savefig(self.data_directory+IMG+f"{self.city}_{geo_div_method}_{cat_div_method}.png",bbox_inches="tight")
                fig.savefig(self.data_directory+IMG+f"{self.city}_{geo_div_method}_{cat_div_method}.eps",bbox_inches="tight")

                if plot_ord:
                    fig = plt.figure(figsize=(8,8))
                    ax = fig.add_subplot(111)
                    val_exp = cat_div_propensity/(cat_div_propensity+geo_div_propensity)
                    ax.plot(100*np.array(self.all_uids)/len(self.all_uids),np.sort(val_exp),color='k')
                    ax.set_xlabel("Users (%)")
                    ax.set_ylabel("$\delta$")
                    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
                    fig.savefig(self.data_directory+IMG+f"{self.city}_ord_{geo_div_method}_{cat_div_method}.png",bbox_inches="tight")
                    fig.savefig(self.data_directory+IMG+f"{self.city}_ord_{geo_div_method}_{cat_div_method}.eps",bbox_inches="tight")

                if plot_cdf:
                    fig = plt.figure(figsize=(8,8))
                    ax = fig.add_subplot(111)
                    num_users = len(self.all_uids)
                    xs = np.linspace(0,1,21)
                    ys = []
                    for x in xs:
                        ys.append(len(np.nonzero(val_exp<=x)[0])/num_users)
                    ax.plot(xs,ys,color='k',marker='o')
                    ax.set_xlabel("x")
                    ax.set_ylabel("P($\delta \leq x)$")
                    fig.savefig(self.data_directory+IMG+f"{self.city}_cdf_{geo_div_method}_{cat_div_method}.png",bbox_inches="tight")
                    fig.savefig(self.data_directory+IMG+f"{self.city}_cdf_{geo_div_method}_{cat_div_method}.eps",bbox_inches="tight")

                if diff_to_optimal:
                    fig = plt.figure(figsize=(8,8))
                    ax = fig.add_subplot(111)
                    ax.plot(self.all_uids,np.sort(cat_div_propensity/(cat_div_propensity+geo_div_propensity) - list(perfect_parameter.values())),color='k')
                    ax.set_xlabel("Users")
                    ax.set_ylabel("$\delta$ diff to optimal")
                    fig.savefig(self.data_directory+IMG+f"{self.city}_ord_diff_{geo_div_method}_{cat_div_method}.png")

    def print_usg_hyperparameter(self):
        KS = [10]
        inl = np.around(np.linspace(0,1,6),2)
        l = []
        for i in inl:
            for j in inl:
                l.append((i,j))
                self.base_rec_parameters['alpha'], self.base_rec_parameters['beta'] = i, j
                self.load_metrics(base=True,name_type=NameType.FULL,METRICS_KS=KS)
        METRIC = 'recall'
        for i,k in enumerate(KS):
            metrics_mean=dict()
            for i,params,metrics in zip(range(len(self.metrics)),l,self.metrics.values()):
                metrics=metrics[k]
                metrics_mean[params]=0
                for obj in metrics:
                    metrics_mean[params]+=obj[METRIC]
                metrics_mean[params]/=len(metrics)
            print(metrics_mean)
            print(pd.Series(metrics_mean).sort_values(ascending=False))



    def plot_usg_hyperparameter(self):
        KS = [10]
        inl = np.around(np.linspace(0,1,6),2)
        l = []
        for i in inl:
            for j in inl:
                for k in inl:
                    l.append((i,j,k))
                    self.base_rec_parameters['alpha'], self.base_rec_parameters['beta'], self.base_rec_parameters['eta'] = i, j, k 
                    self.load_metrics(base=True,name_type=NameType.FULL,METRICS_KS=KS)
        METRIC = 'recall'
        for i,k in enumerate(KS):
            # palette = plt.get_cmap(CMAP_NAME)
            fig = plt.figure(figsize=(16,8))
            ax = plt.axes(projection="3d")
            # ax.grid(alpha=MPL_ALPHA)
            plt.xticks(rotation=90)
            #K = max(experiment_constants.METRICS_K)
            #K = 10
            
            metrics_mean=dict()
            for i,params,metrics in zip(range(len(self.metrics)),l,self.metrics.values()):
                metrics=metrics[k]
                metrics_mean[params]=0
                for obj in metrics:
                    metrics_mean[params]+=obj[METRIC]
                metrics_mean[params]/=len(metrics)
            # for k, v in metrics_mean.items():
            #     print(k,v)
            print(f"at @{k}")
            l_tmp = list(zip(*l))
            p1 = list(zip(*l_tmp[:2]))
            alpha_beta = list(map(lambda pair: "%.2f-%.2f"%(pair[0],pair[1]),p1))
            eta = l_tmp[-1]
            # print(len(alpha_beta),len(eta),)
            xlabels = list(map(lambda pair: "%.2f-%.2f"%(pair[0],pair[1]),list(itertools.product(inl,repeat=2))))
            xticks = np.linspace(0,1,len(xlabels))
            d_label_tick = {label: tick for label, tick in zip(xlabels,xticks)}
            alpha_beta_like_xticks = list(map(d_label_tick.get,alpha_beta))
            print(alpha_beta_like_xticks)
            surf = ax.plot_trisurf(alpha_beta_like_xticks, eta, list(metrics_mean.values()),cmap=plt.get_cmap('Greys'),linewidth=0.1, vmin=0, vmax=np.max(list(metrics_mean.values())))
            fig.colorbar(surf, shrink=0.5, aspect=5)
            ax.scatter(alpha_beta_like_xticks, eta, list(metrics_mean.values()),color='k')
            ax.set(xticks=xticks, xticklabels=xlabels)
            ax.set_xlabel(r"$\alpha$-$\beta$",labelpad=50)
            ax.set_ylabel(r"$\eta$")
            ax.set_zlabel(METRICS_PRETTY[METRIC]+f"@{k}")
            fig.savefig(self.data_directory+IMG+f"{self.city}_{k}_usg_hyperparameter.png")
            # df = pd.Series(metrics_mean)
            # print(df.unstack().unstack())
            # print(df.groupby(level=0).apply(max))
            # print(df.groupby(level=1).apply(max))
            # print(df.groupby(level=2).apply(max))

    def plot_geosoca_hyperparameter(self):
        KS = [10]
        lp = np.around(np.linspace(0, 1, 11),decimals=2)
        l = []
        for alpha in lp:
            l.append(alpha)
            self.base_rec_parameters['alpha'] = alpha
            self.load_metrics(base=True,pretty_name=False,short_name=False,METRICS_KS=KS)
        METRIC = 'recall'
        for i,k in enumerate(KS):
            # palette = plt.get_cmap(CMAP_NAME)
            # fig = plt.figure(figsize=(8,8))
            # ax=fig.add_subplot(111)
            # ax.grid(alpha=MPL_ALPHA)
            # plt.xticks(rotation=90)
            #K = max(experiment_constants.METRICS_K)
            #K = 10
            metrics_mean=dict()
            for i,params,metrics in zip(range(len(self.metrics)),l,self.metrics.values()):
                metrics=metrics[k]
                metrics_mean[params]=0
                for obj in metrics:
                    metrics_mean[params]+=obj[METRIC]
                metrics_mean[params]/=len(metrics)
            print(f"at @{k}")
            df = pd.Series(metrics_mean)
            print(df.sort_values(ascending=False))


    def plot_geocat_hyperparameter(self,metric='maut'):
        KS = [10]
        num_l_cat = 6
        num_l_geocat_div = 5
        l_cat = np.sort(np.append(np.around(np.linspace(0, 1, 6),decimals=2),[0.05,0.1,0.9,0.95]))
        num_l_cat += 4
        l_geocat_div = np.around(np.linspace(0,1,num_l_geocat_div),decimals=2)
        args = []
        l_geocat_div_2 = []
        for div_weight in l_geocat_div:
            self.final_rec_parameters['div_weight'] = div_weight
            for div_geo_cat_weight in l_geocat_div:
                if not(div_weight==0.0 and div_geo_cat_weight!=div_weight):
                    l_geocat_div_2.append((div_weight,div_geo_cat_weight))

                self.final_rec_parameters['div_geo_cat_weight'] = div_geo_cat_weight
                for div_cat_weight in l_cat:
                    self.final_rec_parameters['div_cat_weight'] = div_cat_weight
                    if not(div_weight==0.0 and (div_geo_cat_weight!=div_weight or div_cat_weight!=div_weight)):
                        args.append((div_weight,div_geo_cat_weight,div_cat_weight))
                        self.load_metrics(base=False,name_type=NameType.FULL,METRICS_KS=KS)

        for i,k in enumerate(KS):
            fig = plt.figure(figsize=(10.5,6.3))
            ax = plt.axes(projection="3d")
            font = {'fontsize': 14}
            plt.xticks(rotation=90,**font)
            plt.yticks(**font)
            

            metrics_mean=dict()
            for i,rec_using,metrics in zip(range(len(self.metrics)),self.metrics.keys(),self.metrics.values()):
                metrics=metrics[k]

                metrics_mean[rec_using]=defaultdict(float)
                for obj in metrics:
                    for key in self.metrics_name:
                        metrics_mean[rec_using][key]+=obj[key]


                for j,key in enumerate(metrics_mean[rec_using]):
                    metrics_mean[rec_using][key]/=len(metrics)
            if metric == 'maut':
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

                val_to_use = mauts
            else:
                val_to_use = dict()
                for rec_using, metrics in metrics_mean.items():
                    val_to_use[rec_using] = metrics[metric]

            # val_to_use['phoenix_usg_80_alpha_0_beta_0.2_eta_0_geocat_10_div_weight_0.75_div_geo_cat_weight_0.25_heuristic_local_max_obj_func_cat_weight_div_cat_weight_0.05'],\
            # val_to_use['phoenix_usg_80_alpha_0_beta_0.2_eta_0_geocat_10_div_weight_1.0_div_geo_cat_weight_0.25_heuristic_local_max_obj_func_cat_weight_div_cat_weight_0.0']=\
            # val_to_use['phoenix_usg_80_alpha_0_beta_0.2_eta_0_geocat_10_div_weight_1.0_div_geo_cat_weight_0.25_heuristic_local_max_obj_func_cat_weight_div_cat_weight_0.0'],\
            # val_to_use['phoenix_usg_80_alpha_0_beta_0.2_eta_0_geocat_10_div_weight_0.75_div_geo_cat_weight_0.25_heuristic_local_max_obj_func_cat_weight_div_cat_weight_0.05']
            print(pd.Series(val_to_use).sort_values(ascending=False))
            print(f"at @{k}")
            args_separated = list(zip(*args))
            lambdas = np.array(args_separated[0])
            deltas = np.array(args_separated[1])
            lambda_delta = list(zip(*args_separated[:2]))
            lambda_delta = list(map(lambda pair: "%.2f-%.2f"%(pair[0],pair[1]),lambda_delta))
            phi = args_separated[-1]
            # print(len(alpha_beta),len(eta),)
            xlabels = list(map(lambda pair: "%.2f-%.2f"%(pair[0],pair[1]),l_geocat_div_2))
            xticks = np.linspace(0,1,len(xlabels))
            d_label_tick = {label: tick for label, tick in zip(xlabels,xticks)}
            lambda_delta_like_xticks = list(map(d_label_tick.get,lambda_delta))
            print(lambda_delta_like_xticks)
            surf = ax.plot_trisurf(lambda_delta_like_xticks, phi, list(val_to_use.values()),cmap=plt.get_cmap(CMAP_NAME),linewidth=25, vmin=np.min(list(val_to_use.values())), vmax=np.max(list(val_to_use.values())))
            val_to_use_values = np.array(list(val_to_use.values()))
            id_reorder = np.argsort(-val_to_use_values)
            lambdas = lambdas[id_reorder]
            deltas = deltas[id_reorder]
            phis = np.array(phi)[id_reorder]
            val_to_use_values = val_to_use_values[id_reorder]
            top_n = 5
            i=1
            d_lambda_delta_tick = {label: tick for label, tick in zip(l_geocat_div_2,xticks)}
            if LANG == 'pt':
                box_text_string = "$i$-$\lambda$-$\delta$-$\phi$\n"
            else:
                box_text_string = "$i^{th}$-$\lambda$-$\delta$-$\phi$\n"
            for x,y,z,p in list(zip(lambdas,deltas,phis,val_to_use_values))[:top_n]:
                z_up = (1+0.04*i)*ax.get_zlim()[1]
                # print('$%d^{%s}$-%.2f-%.2f-%.2f'%(i,int_what_ordinal(i),x,y,z))
                x_real = d_lambda_delta_tick[(x,y)]
                if LANG == 'pt':
                    if i == 1:
                        ax.text(x_real,z,z_up,'$\\bf{%d}$'%(i,),fontsize=16,zorder=27)
                        string_to_put = '$\\bf{%d-%.2f-%.2f-%.2f}$'%(i,x,y,z)
                    else:
                        ax.text(x_real,z,z_up,'$%d$'%(i),fontsize=16,zorder=27)
                        string_to_put = '$%d$-%.2f-%.2f-%.2f'%(i,x,y,z)
                else:
                    if i == 1:
                        ax.text(x_real,z,z_up,'$\\bf{%d^{%s}}$'%(i,int_what_ordinal(i)),fontsize=16,zorder=27)
                        string_to_put = '$\\bf{%d^{%s}-%.2f-%.2f-%.2f}$'%(i,int_what_ordinal(i),x,y,z)
                    else:
                        ax.text(x_real,z,z_up,'$%d^{%s}$'%(i,int_what_ordinal(i)),fontsize=16,zorder=27)
                        string_to_put = '$%d^{%s}$-%.2f-%.2f-%.2f'%(i,int_what_ordinal(i),x,y,z)

                box_text_string += string_to_put + '\n'
                if i == 1:
                    a = Arrow3D([x_real, x_real], [z, z], [p, z_up], mutation_scale=20,
                                lw=1, arrowstyle="<|-", color='k',zorder=26)
                else:
                    a = Arrow3D([x_real, x_real], [z, z], [p, z_up], mutation_scale=20,
                                lw=1, arrowstyle="<|-", color='0.55',zorder=26)
                ax.add_artist(a)
                i+=1
            # print(ax.get_xlim()[1], ax.get_ylim()[0],ax.get_zlim()[1]*1.05)
            plt.figtext(0.01,0.65,s=box_text_string[:-1], color='black', fontsize=17,
                    bbox=dict(facecolor='1.0',edgecolor='black', boxstyle='square,pad=0.1'))

            # from mpl_toolkits.axes_grid1 import make_axes_locatable
            # divider = make_axes_locatable(ax)
            # cax = divider.append_axes("right", size="5%",pad=0.05)

            # from mpl_toolkits.axes_grid1 import make_axes_locatable
            # divider = make_axes_locatable(ax)
            # cax = divider.new_vertical(size="5%",pad=0.7,pack_start=True)
            # fig.add_axes(cax)
            # fig.colorbar(surf, cax=cax,orientation='horizontal')
            
            cbaxes = fig.add_axes([0.3, 0.97, 0.4, 0.02]) 
            cbar = fig.colorbar(surf,cbaxes,orientation='horizontal')
            cbar.ax.tick_params(labelsize=font['fontsize'],zorder=25)
            for t in ax.zaxis.get_major_ticks():
                t.label.set_fontsize(font['fontsize'])
            # fig.colorbar(surf, shrink=0.6, aspect=20,
            #              pad= -0.02,orientation="horizontal")
            # fig.colorbar(surf, fraction=0.046, pad=0.04)
            # ax.scatter(lambda_delta_like_xticks, phi, list(val_to_use.values()),cmap=plt.cm.CMRmap,vmin=np.min(list(val_to_use.values())), vmax=np.max(list(val_to_use.values())))
            ax.set(xticks=xticks, xticklabels=xlabels)
            labels_fontsize = 19
            ax.set_xlabel(r"$\lambda$-$\delta$",labelpad=69,fontsize=labels_fontsize)
            ax.set_ylabel(r"$\phi$",labelpad=7,fontsize=labels_fontsize)
            ax.set_zlabel(f"{METRICS_PRETTY[metric]}@{k}",fontsize=labels_fontsize)
            plt.subplots_adjust(left=-0.17,top=1.05,right=1.05)

            fig.savefig(self.data_directory+IMG+f"{self.city}_{k}_{self.base_rec}_geocat_hyperparameter.png")
            fig.savefig(self.data_directory+IMG+f"{self.city}_{k}_{self.base_rec}_geocat_hyperparameter.eps")
            
    def print_latex_cities_metrics_table(self,cities,prefix_name='',references=[],heuristic=False,print_htime=False,recs=['gc','ld','geodiv','geocat']):
        num_cities = len(cities)
        num_metrics = len(self.metrics_name)
        result_str = Text()
        result_str += r"\begin{table}[]\scriptsize"
        if heuristic and print_htime:
            num_metrics += 1
        ls_str = ''.join([('l'*num_metrics+'|') if i!=(num_cities-1) else ('l'*num_metrics) for i in range(num_cities)])
        result_str += r"\begin{tabular}{" +'l|'+ls_str + "}"
        line_start = len(result_str)

        final_rec_list_size = self.final_rec_list_size

        for city in cities:
            self.city = city
            self.final_rec_list_size = final_rec_list_size
            self.load_metrics(base=True,name_type=NameType.PRETTY)
            
            run_times = dict()
            # if heuristic:
            #     run_times[self.get_base_rec_pretty_name()] = open(self.data_directory+UTIL+f'run_time_{self.get_base_rec_name()}.txt',"r").read()
            
            if not heuristic:
                for rec in recs:
                    self.final_rec = rec
                    self.load_metrics(base=False,name_type=NameType.PRETTY)

            if heuristic:
                self.final_rec_parameters = {'heuristic': 'local_max'}
                self.load_metrics(base=False,name_type=NameType.PRETTY)

            if not references:
                references = [list(self.metrics.keys())[0]]
            line_num = line_start
            if city != cities[-1]:
                result_str[line_num] += "\t&"+'\multicolumn{%d}{c|}{%s}' % (num_metrics,CITIES_PRETTY[city])
            else:
                result_str[line_num] += "\t&"+'\multicolumn{%d}{c}{%s}' % (num_metrics,CITIES_PRETTY[city])
            if city == cities[-1]:
                result_str[line_num] += '\\\\'
            line_num += 1
            for i,k in enumerate(experiment_constants.METRICS_K):

                if heuristic:
                    self.final_rec_list_size = k
                    for h in ['tabu_search', 'particle_swarm']:
                        self.final_rec_parameters = {'heuristic': h}
                        self.load_metrics(base=False,name_type=NameType.PRETTY,METRICS_KS=[k])
                        run_times[self.get_final_rec_pretty_name()]=int(float(open(self.data_directory+UTIL+f'run_time_{self.get_final_rec_name()}.txt',"r").read()))
                    self.final_rec_parameters = {'heuristic': 'local_max'}
                    run_times[self.get_final_rec_pretty_name()]=int(float(open(self.data_directory+UTIL+f'run_time_{self.get_final_rec_name()}.txt',"r").read()))
                    max_arg_run_time = max(run_times, key=run_times.get)


                if len(result_str[line_num]) == 0:
                    result_str[line_num] += "\\hline \\rowcolor{Gray} \\textbf{Algorithm} & "+'& '.join(map(lambda x: "\\textbf{"+METRICS_PRETTY[x]+f"@{k}}}" ,self.metrics_name))
                else:
                    result_str[line_num] += '& '+ '& '.join(map(lambda x: "\\textbf{"+METRICS_PRETTY[x]+f"@{k}}}" ,self.metrics_name))
                if heuristic and print_htime:
                    result_str[line_num] += '& \\textbf{Time}'
                if city == cities[-1]:
                    result_str[line_num] += "\\\\"
                line_num += 1
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
                                metrics_gain[rec_using][metric_name] = bullet_str
                            else:
                                if metrics_mean[rec_using][metric_name] < metrics_mean[reference_name][metric_name]:
                                    metrics_gain[rec_using][metric_name] = triangle_down_str
                                elif metrics_mean[rec_using][metric_name] > metrics_mean[reference_name][metric_name]:
                                    metrics_gain[rec_using][metric_name] = triangle_up_str
                                else:
                                    metrics_gain[rec_using][metric_name] = bullet_str 
                    else:
                        reference_name = rec_using
                        base_metrics = metrics
                        metrics_gain[rec_using] = dict()
                        for metric_name in self.metrics_name:
                            metrics_gain[rec_using][metric_name] = ''
                metrics_max = {mn:(None,0) for mn in self.metrics_name}
                for rec_using,rec_metrics in metrics_mean.items():
                    for metric_name, value in rec_metrics.items():
                        if metrics_max[metric_name][1] < value:
                            metrics_max[metric_name] = (rec_using,value)
                is_metric_the_max = defaultdict(lambda: defaultdict(bool))
                for metric_name,(rec_using,value) in metrics_max.items():
                    is_metric_the_max[rec_using][metric_name] = True

                for rec_using,rec_metrics in metrics_mean.items():
                    gain = metrics_gain[rec_using]
                    
                    if city == cities[0]:
                        result_str[line_num] += rec_using
                    result_str[line_num] += ' &' + '& '.join(map(lambda x: "\\textbf{%.4f}%s" %(x[0],x[1]) if is_metric_the_max[rec_using][x[2]] else "%.4f%s"%(x[0],x[1])  ,zip(rec_metrics.values(),gain.values(),rec_metrics.keys())))
                    if heuristic and print_htime:
                        if rec_using not in references:
                            if rec_using != max_arg_run_time:
                                result_str[line_num] += f'& {sec_to_hm(run_times[rec_using])}'
                            else:
                                result_str[line_num] += f'& \\textbf{{{sec_to_hm(run_times[rec_using])}}}'
                        else:
                            result_str[line_num] += f'& '
                            
                    if city == cities[-1]:
                        result_str[line_num] += "\\\\"
                    line_num += 1

        result_str += "\\end{tabular}"
        result_str += "\\end{table}"
        result_str = result_str.__str__()
        result_str = LATEX_HEADER+result_str
        result_str += LATEX_FOOT
        fout = open(self.data_directory+UTIL+'_'.join(references)+'_'+'side_'+'_'.join(([prefix_name] if len(prefix_name)>0 else [])+cities)+'.tex', 'w')
        fout.write(result_str)
        fout.close()

    def gc(self):
        args=[(uid,) for uid in self.all_uids]
        results = run_parallel(self.run_gc,args,self.CHKS)
        self.save_result(results,base=False)

    @classmethod
    def run_gc(cls, uid):
        self = cls.getInstance()
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

    def print_div_weight_hyperparameter(self,lp=np.around(np.linspace(0, 1, 11),decimals=2)):
        KS = [10]
        # lp = np.around(np.linspace(0, 1, 11),decimals=2)
        l = []
        for div_weight in lp:
            l.append(div_weight)
            self.final_rec_parameters['div_weight'] = div_weight
            self.load_metrics(base=False,name_type = NameType.FULL,METRICS_KS=KS)

        for i,k in enumerate(KS):
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

            print(pd.Series(mauts).sort_values(ascending=False))


    def plot_adapt_gain_metrics(self,cities,base_recs,final_recs,prefix_name='all_met_gain_cities',ncol=3):
        idx = 0
        refs_idxs = [] 
        for city in cities:
            self.city = city
            for base_rec in base_recs:
                self.base_rec = base_rec
                self.load_metrics(base=True,name_type=NameType.FULL)
                refs_idxs.append(idx)
                idx += 1
                for final_rec in final_recs:
                    self.final_rec = final_rec
                    self.load_metrics(base=False,name_type=NameType.CITY_BASE_PRETTY)
                    idx += 1
            
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

            reference_recommenders = [list(metrics_mean.keys())[idx] for idx in refs_idxs]
            reference_vals = [np.array(list(metrics_mean.pop(ref_rec).values())) for ref_rec in reference_recommenders]
            fig = plt.figure()
            ax=fig.add_subplot(111)
            ax.grid(alpha=MPL_ALPHA)
            num_recs_plot = len(metrics_mean)
            barWidth= 1-num_recs_plot/(1+num_recs_plot)
            N=len(self.metrics_name)
            indexes=np.arange(N)
            i=0
            styles = gen_bar_cycle(len(self.metrics))()
            for rec_using,rec_metrics in metrics_mean.items():
                print(f"{rec_using} at @{k}")
                print(rec_metrics)
                
                ax.bar(indexes+i*barWidth,-100+100*np.array(list(rec_metrics.values()))/reference_vals[i//len(final_recs)],barWidth,label=rec_using,**next(styles))
                i+=1
            ax.set_xticks(np.arange(N+1)+barWidth*(((num_recs_plot)/2)-1)+barWidth/2)
            ax.legend(tuple(metrics_mean.keys()),bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                      mode="expand", borderaxespad=0, ncol=ncol)
            ax.set_xticklabels(list(map(lambda name: METRICS_PRETTY[name],self.metrics_name)))
            ax.set_ylabel("Gain after the reordering (%)")
            ax.set_xlim(-barWidth,len(self.metrics_name)-1+(num_recs_plot-1)*barWidth+barWidth)
            fig.show()
            plt.show()
            timestamp = datetime.timestamp(datetime.now())
            fig.savefig(self.data_directory+f"result/img/{prefix_name}_{'_'.join(base_recs)}_{self.city}_{str(k)}.png",bbox_inches="tight")

    def plot_jaccard_ild_gc_correlation(self,base_rec,final_rec_1,final_rec_2,ks,metrics=['ild','gc']):
        lambda_used = 1
        self.final_rec = final_rec_1
        self.final_rec_parameters['div_weight'] = lambda_used
        div_weight_1 = self.final_rec_parameters['div_weight']
        self.load_final_predicted()
        res_1 = self.user_final_predicted_lid
        
        
        self.final_rec = final_rec_2
        self.final_rec_parameters['div_weight'] = lambda_used
        div_weight_2 = self.final_rec_parameters['div_weight']
        self.load_final_predicted()
        res_2 = self.user_final_predicted_lid

        self.final_rec = 'random'
        self.final_rec_parameters = {'div_weight': div_weight_1}
        self.load_final_predicted()
        res_3 = self.user_final_predicted_lid
        self.final_rec_parameters = {'div_weight': div_weight_2+0.00001}
        self.load_final_predicted()
        res_4 = self.user_final_predicted_lid
        
        values = dict()
        values_rand = dict()
        # spearmans = dict()
        for k in ks:
            num_div = 0
            num_equal = 0
            # spearman_acc = 0
            equal_pois_data = np.array([])
            # spearman_data = np.array([])

            for (uid1, lids1), (uid2,lids2) in zip(res_1.items(),res_2.items()):
                equal_pois = set(lids1[:k]).intersection(set(lids2[:k]))
                # ranks_1= [lids1.index(lid) for lid in equal_pois]
                # ranks_2= [lids2.index(lid) for lid in equal_pois]
                equal_pois_data = np.append(equal_pois_data,len(equal_pois))
                # if len(equal_pois) > 1:
                #     spearman = scipy.stats.spearmanr(ranks_1,ranks_2)[0]
                #     spearman_data = np.append(spearman_data,spearman)
                    
                num_equal += len(equal_pois)
                num_div += k
            print(f'At @{k}')
            print("Equal POIs")
            print(scipy.stats.describe(equal_pois_data))
            values[k] = num_equal/num_div
            equal_pois_rand = np.array([])
            for (uid1, lids1), (uid2,lids2) in zip(res_3.items(),res_4.items()):
                equal_pois = set(lids1[:k]).intersection(set(lids2[:k]))
                equal_pois_rand = np.append(equal_pois_rand,len(equal_pois))

            values_rand[k] = equal_pois_rand.sum()/(k*len(res_1))
            # if len(spearman_data) > 0:
            #     print("Spearman in Equal POIs")
            #     print(scipy.stats.describe(spearman_data))
            #     spearmans[k] = np.mean(spearman_data)

        plt.rcParams.update({'font.size': 19})
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(list(values.keys()),list(values.values()),marker='o',color='k')
        ax.plot(list(values_rand.keys()),list(values_rand.values()),marker='o',color='r')
        ax.legend((f'{RECS_PRETTY[final_rec_1]}($\lambda$={div_weight_1}),{RECS_PRETTY[final_rec_2]}($\lambda$={div_weight_2})',f'Random($\lambda$={div_weight_1}),Random($\lambda$={div_weight_2})'),frameon = False)
        ax.set_ylim(0,1)
        if LANG == 'pt':
            ax.set_xlabel('Tamanho da lista')
            ax.set_ylabel('Taxa de POIs iguais')
        else:
            ax.set_xlabel('List size')
            ax.set_ylabel('Rate of equal POIs')
        xticks = np.round(np.linspace(ks[0],ks[-1],6))
        ax.set_xticks(xticks)

        # ax = fig.add_subplot(122)
        # ax.plot(list(spearmans.keys()),list(spearmans.values()),marker='o',color='k')
        # ax.set_ylim(0,1)
        # ax.set_xlabel('List size')
        # ax.set_ylabel('Correlation coeff.')

        fig.savefig(self.data_directory+IMG+f"{final_rec_1}_{final_rec_2}_{self.city}_{ks[0]}_{ks[-1]}.png",bbox_inches="tight")
        fig.savefig(self.data_directory+IMG+f"{final_rec_1}_{final_rec_2}_{self.city}_{ks[0]}_{ks[-1]}.eps",bbox_inches="tight")

    def plot_ild_gc_correlation(self,metrics=['ild','gc']):
        rec = list(self.metrics.keys())[-1]
        metrics_ks = list(self.metrics.values())[-1]
        correlations = dict()
        for k, metrics_k in metrics_ks.items():
            print("%s@%d correlation"%(rec,k))
            df_p_metrics = pd.DataFrame(metrics_k)
            df_p_metrics = df_p_metrics.set_index('user_id')
            correlations[k] = abs(df_p_metrics[metrics].corr().loc['ild','gc'])
        ks = list(metrics_ks.keys())
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(list(correlations.keys()),list(correlations.values()),marker='o',color='k')
        ax.set_ylim(min(0,min(correlations.values())),1)
        ax.set_xlabel(f'List size ({rec})')
        ax.set_ylabel('Pearson correlation(Absolute)')
        xticks = np.round(np.linspace(ks[0],ks[-1],6))
        ax.set_xticks(xticks)
        fig.savefig(self.data_directory+IMG+f"correlation_{rec}_{self.city}.png",bbox_inches="tight")
        fig.savefig(self.data_directory+IMG+f"correlation_{rec}_{self.city}.eps",bbox_inches="tight")

    def plot_ild_gc_axis(self,metrics=['gc','ild']):
        rec = list(self.metrics.keys())[-1]
        metrics_ks = list(self.metrics.values())[-1]
        k=10
        # print("%s@%d correlation"%(rec,k))
        df_p_metrics = pd.DataFrame(metrics_ks[10])
        df_p_metrics = df_p_metrics.set_index('user_id')

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(df_p_metrics[metrics[0]],df_p_metrics[metrics[1]],color='k')
        ax.set_xlabel(f'{METRICS_PRETTY[metrics[0]]}@{k}')
        ax.set_ylabel(f'{METRICS_PRETTY[metrics[1]]}@{k}')
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        fig.savefig(self.data_directory+IMG+f"correlation_scatter_{rec}_{self.city}.png",bbox_inches="tight")
        fig.savefig(self.data_directory+IMG+f"correlation_scatter_{rec}_{self.city}.eps",bbox_inches="tight")

    
    def dbscan_hyperparameter(self):
        num_users = self.training_matrix.shape[0]
        vals = np.linspace(0.1,5,50)
        mins_samples = np.arange(2,11,1)
        users_visits = np.sum(self.training_matrix,axis=1)
        users_sorted_by_visits = list(reversed(np.argsort(users_visits)))
        users_X = dict()
        for uid in users_sorted_by_visits[:num_users]:
            users_X[uid] = [self.poi_coos[lid] for lid in np.nonzero(self.training_matrix[uid])[0]]
            print('uid:',uid,'pois:',len(users_X[uid]))

        max_silhouette = 0
        max_val = 0
        max_min_samples = 0
        all_max_silhouette = []
        for val in vals:
            for min_samples in mins_samples:

                users_sil = np.array([])
                for uid in users_sorted_by_visits[:num_users]:
                    X = users_X[uid]
                    db = DBSCAN(eps=geo_utils.km_to_lat(val), min_samples=min_samples).fit(X)
                    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
                    core_samples_mask[db.core_sample_indices_] = True
                    labels = db.labels_
                    # plt.scatter(x=X[:,0],y=X[:,1],c=labels)
                    try:
                        sil = silhouette_score(X, labels)
                        users_sil = np.append(users_sil,sil)
                    except:
                        pass
                    # print(f'{val} {min_samples}',"Silhouette Coefficient: %0.3f"
                    #     % sil)
                if np.mean(users_sil) > max_silhouette:
                    max_silhouette = sil
                    max_val = val
                    max_min_samples = min_samples
                    all_max_silhouette.append((max_silhouette, max_val, max_min_samples))
        print(all_max_silhouette)
            


    def random(self):
        args=[(uid,) for uid in self.all_uids]
        results = run_parallel(self.run_random,args,self.CHKS)
        self.save_result(results,base=False)

    @classmethod
    def run_random(cls, uid):
        self = cls.getInstance()
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

    def plot_diff_to_perfect(self):
        bckp = vars(self).copy()

        self.final_rec = 'perfectpgeocat'
        fin = open(self.data_directory+UTIL+f'parameter_{self.get_final_rec_name()}.pickle',"rb")
        self.perfect_parameter = pickle.load(fin)

        for key, val in bckp.items():
            vars(self)[key] = val

        self.persongeocat_preprocess()
        div_geo_cat_weight = self.div_geo_cat_weight
        # print(self.perfect_parameter)
        vals = np.array([])
        for uid, val in self.erfect_parameter.items():
            vals = np.append(vals,val)
        diff = div_geo_cat_weight-vals   
        unique, counts = np.unique(diff, return_counts=True)
        print(dict(zip(unique, counts)))

        lab_enc = preprocessing.LabelEncoder()
        y_test = lab_enc.fit_transform(vals)
        pred = lab_enc.transform(div_geo_cat_weight)

        print(classification_report(y_test, pred,zero_division=0))
        print(confusion_matrix(y_test, pred))
        # plt.plot(diff,color='k')

        # plt.savefig(self.data_directory+IMG+f'perf_param_diff_{self.get_final_rec_name()}.png')

    def plot_heuristics_maut(self,prefix_name='maut',ncol=2):
        fig = plt.figure()
        ax=fig.add_subplot(111)
        num_recs = 4
        N=len(experiment_constants.METRICS_K)
        barWidth=1-num_recs/(1+num_recs)
        indexes=np.arange(N)

        run_times = dict()
        print(self.get_base_rec_name())
        self.load_metrics(base=True,name_type=NameType.PRETTY)
        self.final_rec_parameters = {'heuristic': 'local_max'}
        self.load_metrics(base=False,name_type=NameType.PRETTY)

        rects = dict()
        for count,k in enumerate(experiment_constants.METRICS_K):
            self.final_rec_list_size = k
            for h in ['tabu_search', 'particle_swarm']:
                self.final_rec_parameters = {'heuristic': h}
                self.load_metrics(base=False,name_type=NameType.PRETTY,METRICS_KS=[k])
                run_times[self.get_final_rec_pretty_name()]=int(float(open(self.data_directory+UTIL+f'run_time_{self.get_final_rec_name()}.txt',"r").read()))
            self.final_rec_parameters = {'heuristic': 'local_max'}
            run_times[self.get_final_rec_pretty_name()]=int(float(open(self.data_directory+UTIL+f'run_time_{self.get_final_rec_name()}.txt',"r").read()))
            # max_arg_run_time = max(run_times, key=run_times.get)

            metrics_mean=dict()
            for i,rec_using,metrics in zip(range(len(self.metrics)),self.metrics.keys(),self.metrics.values()):
                metrics=metrics[k]

                metrics_mean[rec_using]=defaultdict(float)
                for obj in metrics:
                    for key in self.metrics_name:
                        metrics_mean[rec_using][key]+=obj[key]


                for j,key in enumerate(metrics_mean[rec_using]):
                    metrics_mean[rec_using][key]/=len(metrics)
            styles = gen_bar_cycle(len(self.metrics),ord_by_brightness=True)()

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
            i=0
            for rec_using,utility_scores in metrics_utility_score.items():
                val = mauts[rec_using]
                rects[rec_using] = ax.bar(count+i*barWidth,val,barWidth,**next(styles),label=rec_using)[0]
                    # ax.bar(count+i*barWidth,val,barWidth,**GEOCAT_BAR_STYLE,label=rec_using)
                # if print_text:
                if rec_using != 'USG':
                    ax.text(count+i*barWidth-barWidth/2+barWidth*0.1,val-0.225,"%s"%(sec_to_hm(run_times[rec_using])),rotation=90,zorder=33,fontsize=14,fontweight='bold')
                i+=1
            #ax.set_xticks(np.arange(N+1)+barWidth*(np.floor((len(self.metrics))/2)-1)+barWidth/2)
        # ax.set_xticks(np.arange(N+1)+barWidth*len(metrics_utility_score)/2+barWidth/2)

        rects = list(rects.values())

        ax.set_xticks(np.arange(N+1)+barWidth*(((len(self.metrics))/2)-1)+barWidth/2)
        # ax.legend((p1[0], p2[0]), self.metrics_name)
        ax.legend(rects,tuple(self.metrics.keys()),bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                  mode="expand", borderaxespad=0, ncol=ncol,
                  handler_map={rect: HandlerSquare() for rect in rects},
                  fontsize=15,handletextpad=-0.6)
        # ax.legend(tuple(map(lambda name: METRICS_PRETTY[name],self.metrics.keys())))
        ax.set_xticklabels(['@5','@10','@20'],fontsize=16)
        # ax.set_title(f"at @{k}, {self.city}")
        ax.set_ylabel("MAUT",fontsize=18)
        ax.set_ylim(0,1)
        ax.set_xlim(-barWidth,len(experiment_constants.METRICS_K)-1+(len(self.metrics)-1)*barWidth+barWidth)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(16)
                  

        for i in range(N):
            ax.axvline(i+num_recs*barWidth,color='k',linewidth=1,alpha=0.5)
        fig.show()
        plt.show()

        timestamp = datetime.timestamp(datetime.now())
        fig.savefig(self.data_directory+f"result/img/{prefix_name}_{self.base_rec}_{self.city}.png",bbox_inches="tight")
        fig.savefig(self.data_directory+f"result/img/{prefix_name}_{self.base_rec}_{self.city}.eps",bbox_inches="tight")


    def plot_gain_metric(self,prefix_name='all_met_gain',metrics_name=None):
        if metrics_name is None:
            metrics_name = self.metrics_name
        # palette = plt.get_cmap(CMAP_NAME)
        for i,k in enumerate(experiment_constants.METRICS_K):
            metrics_mean=dict()
            for i,rec_using,metrics in zip(range(len(self.metrics)),self.metrics.keys(),self.metrics.values()):
                metrics=metrics[k]
                
                metrics_mean[rec_using]=defaultdict(float)
                for obj in metrics:
                    for key in metrics_name:
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
                val = np.sum(list(utility_scores.values()))/len(utility_scores)
                mauts[rec_using] = val
                metrics_mean[rec_using]['maut'] = val

            reference_recommender = list(metrics_mean.keys())[0]
            reference_vals = metrics_mean.pop(reference_recommender)
            for metric_name in metrics_name+['maut']:
                fig = plt.figure(figsize=(8, 6))
                ax=fig.add_subplot(111)
                ax.grid(alpha=MPL_ALPHA,axis='y')
                num_recs_plot = len(self.metrics)-1
                barWidth= 1-num_recs_plot/(1+num_recs_plot)
                i=0
                styles = gen_bar_cycle(num_recs_plot)()
                y_to_annotate = 90
                put_annotation = []
                for rec_using,rec_metrics in metrics_mean.items():
                    print(f"{rec_using} at @{k}")
                    print(rec_metrics)
                    rel_diffs = -100+100*rec_metrics[metric_name]/reference_vals[metric_name]
                    ax.bar(i*barWidth,rel_diffs,barWidth,label=rec_using,**next(styles))
                    # special_cases = rel_diffs > 100
                    # for idx, value in zip(indexes[special_cases],rel_diffs[special_cases]):
                    #     ax.annotate(f'{int(value)}%',xy=(idx+i*barWidth-barWidth,y_to_annotate),zorder=25,color='k',fontsize=18)
                    #     y_to_annotate -= 10
                    i+=1
                ax.set_xticks(np.arange(1)+barWidth*(((num_recs_plot)/2)-1)+barWidth/2)
                # ax.legend((p1[0], p2[0]), metrics_name_)
                ax.legend(tuple(metrics_mean.keys()),bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                          mode="expand", borderaxespad=0, ncol=4,fontsize=7)
                # ax.legend(tuple(map(lambda name: METRICS_PRETTY[name],self.metrics.keys())))
                ax.set_xticklabels([f"{metric_name}@{k}"],fontsize=16)
                for tick in ax.yaxis.get_major_ticks():
                    tick.label.set_fontsize(16) 
                ax.set_ylabel(f"Relative diff w.r.t. {reference_recommender}",fontsize=23)
                # ax.set_ylim(-25,100)
                ax.set_xlim(-barWidth,(num_recs_plot-1)*barWidth+barWidth)

                timestamp = datetime.timestamp(datetime.now())
                fig.savefig(self.data_directory+f"result/img/{prefix_name}_{self.base_rec}_{self.city}_{metric_name}_{str(k)}.png",bbox_inches="tight")
                # fig.savefig(self.data_directory+f"result/img/{prefix_name}_{self.base_rec}_{self.city}_{metric_name}_{str(k)}.eps",bbox_inches="tight")
    def cluster_methods(self):

        met2 = 'poi_ild'
        met3 = 'walk'
        
        self.final_rec = 'persongeocat'
        self.pcat_div_runner = CatDivPropensity.getInstance(
            self.training_matrix,
            self.undirected_category_tree,
            'num_cat',
            self.poi_cats)
        num_cat=self.pcat_div_runner.compute_div_propensity()

        self.pcat_div_runner = CatDivPropensity.getInstance(
            self.training_matrix,
            self.undirected_category_tree,
            met2,
            self.poi_cats)
        ild=self.pcat_div_runner.compute_div_propensity()

        self.pgeo_div_runner = GeoDivPropensity.getInstance(self.training_matrix, self.poi_coos,
                                                            self.poi_cats,self.undirected_category_tree,
                                                            met3)
        walk = self.pgeo_div_runner.compute_div_propensity()
        fig = plt.figure()
        ax=fig.add_subplot(111)
        ax.hist(num_cat,bins=80)
        fig.savefig(self.data_directory+f"result/img/dist_num_cat_{self.city}.png",bbox_inches="tight")
        print('num_cat: ', scipy.stats.describe(num_cat))
        print(f'{met2}: ',scipy.stats.describe(ild))
        print(f'{met3}: ',scipy.stats.describe(walk))
        fig = plt.figure(figsize=(16,8))
        ax = plt.axes(projection="3d")
        ax.scatter(num_cat,ild,walk)
        ax.set_xlabel('num_cat')
        ax.set_ylabel(met2)
        ax.set_zlabel(met3)
        fig.savefig(self.data_directory+f"result/img/cluster_methods_{self.city}.png",bbox_inches="tight")


    def print_base_info(self):
        val= list(map(len,self.poi_cats.values()))
        print("POI Categories")
        print(scipy.stats.describe(val))
        print('median',np.median(val))

        print("Visits")
        val= np.sum(self.training_matrix,axis=1)
        print(scipy.stats.describe(val))
        print('median',np.median(val))

        users_categories_visits = cat_utils.get_users_cat_visits(self.training_matrix,
                                                                 self.poi_cats)
        print('users_categories_visits')
        print(scipy.stats.describe(list(map(len,users_categories_visits))))
        print(np.median(list(map(len,users_categories_visits))))


    def plot_base_info(self):

        fig = plt.figure()
        ax=fig.add_subplot(111)
        val= list(map(len,self.poi_cats.values()))
        ax.hist(val,bins=len(np.unique(val)),color='k')
        fig.savefig(self.data_directory+f"result/img/poi_num_cat_{self.city}.png",bbox_inches="tight")
        print("poi_num_cat")
        print(scipy.stats.describe(val))
        print('median', np.median(val))
        fig = plt.figure()
        ax=fig.add_subplot(111)
        val= np.sum(self.training_matrix,axis=1)

        print("visits")
        print(scipy.stats.describe(val))
        print('median', np.median(val))
        ax.hist(val,bins=80,color='k')
        fig.savefig(self.data_directory+f"result/img/visits_{self.city}.png",bbox_inches="tight")

        pass

    # Plot mean of groups, and groups by file
    def plot_person_metrics_groups(self,prefix_name='person_groups',ncol=3):
        # self.metrics_name = ['precision','recall']
        self.final_rec_parameters = {'cat_div_method': 'inv_num_cat', 'geo_div_method': 'walk', 'norm_method': 'quadrant','bins': None,'funnel':None}
        # self.final_rec_parameters = {'cat_div_method': 'inv_num_cat', 'geo_div_method': 'walk', 'norm_method': 'default','bins': None,'funnel':None}
        # self.load_metrics(base=False,name_type=NameType.FULL)
        uid_group = self.get_groups()
        groups_count = Counter(uid_group.values())
        unique_groups = list(groups_count.keys())
        palette = plt.get_cmap(CMAP_NAME)

        plt.rcParams.update({'font.size': 14})
        for i,k in enumerate(experiment_constants.METRICS_K):
            metrics_mean=dict()

            groups_mean = dict()
            for i,rec_using,metrics in zip(range(len(self.metrics)),self.metrics.keys(),self.metrics.values()):
                metrics=metrics[k]
                
                metrics_mean[rec_using]=defaultdict(lambda: defaultdict(float))
                groups_mean[rec_using] = defaultdict(float)
                for obj in metrics:
                    group = uid_group[obj['user_id']]
                    for key in self.metrics_name:
                        metrics_mean[rec_using][group][key]+=obj[key]
                
                for j,key in enumerate(self.metrics_name):
                    for group in unique_groups:
                        metrics_mean[rec_using][group][key]/=groups_count[group]

                for j,key in enumerate(self.metrics_name):
                    groups_mean[rec_using][key] = 0
                    for group in unique_groups:
                        groups_mean[rec_using][key] += metrics_mean[rec_using][group][key]
                    groups_mean[rec_using][key] /= len(unique_groups)

            reference_recommender = list(metrics_mean.keys())[0]
            reference_vals = metrics_mean.pop(reference_recommender)

            group_reference_recommender = list(groups_mean.keys())[0]
            group_reference_vals = groups_mean.pop(reference_recommender)

            fig = plt.figure()
            ax=fig.add_subplot(111)
            # ax.grid(alpha=MPL_ALPHA)
            num_recs = len(groups_mean)
            barWidth= 1-num_recs/(1+num_recs)
            N=len(self.metrics_name)
            indexes=np.arange(N)
            i=0
            styles = gen_bar_cycle(num_recs)()

            for rec_using,rec_metrics in groups_mean.items():
                group_metrics = np.array(list(rec_metrics.values()))
                tmp_ref_val = np.array(list(group_reference_vals.values()))
                vals = 100*(group_metrics - tmp_ref_val)/tmp_ref_val
                ax.bar(indexes+i*barWidth,vals,barWidth,label=rec_using,**next(styles))
                i+=1

            ax.set_xticks(np.arange(N+1)+barWidth*(((num_recs)/2)-1)+barWidth/2)
            ax.legend(tuple(groups_mean.keys()),bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                    mode="expand", borderaxespad=0, ncol=ncol)
            ax.set_xticklabels(map(lambda x: f'{x}@{k}',list(map(lambda name: METRICS_PRETTY[name],self.metrics_name))))
            ax.set_ylabel(f"Relative diff w.r.t. {group_reference_recommender}")
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())
            ax.set_xlim(-barWidth,len(self.metrics_name)-1+(num_recs-1)*barWidth+barWidth)
            for i in range(N):
                ax.axvline(i+num_recs*barWidth,color='k',linewidth=1,alpha=0.5)
            fig.savefig(self.data_directory+f"result/img/{prefix_name}_{self.base_rec}_{self.city}_{str(k)}.png",bbox_inches="tight")

            for group in unique_groups:
                fig = plt.figure()
                ax=fig.add_subplot(111)
                # ax.grid(alpha=MPL_ALPHA)
                num_recs = len(metrics_mean)
                barWidth= 1-num_recs/(1+num_recs)
                N=len(self.metrics_name)
                indexes=np.arange(N)
                i=0
                styles = gen_bar_cycle(num_recs)()

                for rec_using,rec_metrics in metrics_mean.items():
                    print(group)
                    group_metrics = np.array(list(rec_metrics[group].values()))
                    # print(f"{rec_using} at @{k}")
                    # print(rec_metrics)
                    tmp_ref_val = np.array(list(reference_vals[group].values()))
                    # print(tmp_ref_val)
                    vals = 100*(group_metrics - tmp_ref_val)/tmp_ref_val
                    # vals= group_metrics
                    ax.bar(indexes+i*barWidth,vals,barWidth,label=rec_using,**next(styles))
                    i+=1

                ax.set_xticks(np.arange(N+1)+barWidth*(((num_recs)/2)-1)+barWidth/2)
                ax.legend(tuple(metrics_mean.keys()),bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                        mode="expand", borderaxespad=0, ncol=ncol)
                ax.set_xticklabels(map(lambda x: f'{x}@{k}',list(map(lambda name: METRICS_PRETTY[name],self.metrics_name))))
                ax.set_ylabel(f"Relative diff w.r.t. {reference_recommender}")
                ax.yaxis.set_major_formatter(mtick.PercentFormatter())
                ax.annotate(f'{group}',xy=(0,0.02),
                                xycoords='axes fraction')
                ax.set_xlim(-barWidth,len(self.metrics_name)-1+(num_recs-1)*barWidth+barWidth)
                for i in range(N):
                    ax.axvline(i+num_recs*barWidth,color='k',linewidth=1,alpha=0.5)
                fig.savefig(self.data_directory+f"result/img/{prefix_name}_{group}_{self.base_rec}_{self.city}_{str(k)}.png",bbox_inches="tight")

    def print_train_perfect(self):
        self.load_perfect()
        a = list(self.perfect_parameter.values())
        perf1 = self.perfect_parameter
        unique, counts = np.unique(a, return_counts=True)
        print(dict(zip(unique, counts)))
        b= [uid for uid in self.all_uids if uid not in self.perfect_parameter]
        print(b)

        self.final_rec_parameters['train_size'] = None
        self.load_perfect()
        perf2 = self.perfect_parameter
        a = list(self.perfect_parameter.values())
        a = set(np.nonzero(a)[0])
        inters = a & set(b)
        print(f'number of equals(a&b): {len(inters)}, a length: {len(a)}, b length: {len(b)}')
        print('intersection:',inters)


        for uid in self.all_uids:
            if uid not in perf1:
                perf1[uid] = 0.5
            
        # perf2 = {key: val for key, val in zip(perf1.keys(),perf2.values())}

        lab_enc = preprocessing.LabelEncoder()

        print(confusion_matrix(lab_enc.fit_transform(list(perf2.values())),
                               lab_enc.transform(list(perf1.values()))))

    def get_groups(self):
        fgroups = open(self.data_directory+UTIL+f'groups_{self.get_final_rec_name()}.pickle','rb')
        groups=pickle.load(fgroups)
        fgroups.close()
        return groups


    def plot_cities_cdf(self):
        plt.rcParams.update({'font.size': 23})
        cities = ['lasvegas','phoenix']
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        colors = ord_scheme_brightness(MY_COLOR_SCHEME)[:len(cities)]
        xs = np.linspace(0,1,21)
        for city, color in zip(cities,colors):
            self.city = city
            self.load_base()
            self.persongeocat_preprocess()
            val_exp = self.div_geo_cat_weight
            num_users = len(self.all_uids)
            ys = []
            for x in xs:
                ys.append(len(np.nonzero(val_exp<=x)[0])/num_users)
            ax.plot(xs,ys,color=color,marker='o')
        ax.legend(list(map(CITIES_PRETTY.get,cities)),frameon=False)
        ax.set_xlabel("x")
        ax.set_ylabel("P($\delta \leq x)$")
        fig.savefig(self.data_directory+IMG+f"{'_'.join(cities)}_cdf_{self.final_rec_parameters['geo_div_method']}_{self.final_rec_parameters['cat_div_method']}.png",bbox_inches="tight")
        fig.savefig(self.data_directory+IMG+f"{'_'.join(cities)}_cdf_{self.final_rec_parameters['geo_div_method']}_{self.final_rec_parameters['cat_div_method']}.eps",bbox_inches="tight")

    # Plot all groups in one file, ks by file
    def plot_metrics_groups(self,prefix_name='groups_metrics',ncol=2):
        self.final_rec_parameters = {'cat_div_method': 'inv_num_cat', 'geo_div_method': 'walk', 'norm_method': 'quadrant','bins': None,'funnel':None}
        uid_group = self.get_groups()
        groups_count = Counter(uid_group.values())
        unique_groups = list(groups_count.keys())
        unique_groups = [x for y, x in sorted(zip(iter(map(GROUP_ID.get,unique_groups)), unique_groups))]
        palette = plt.get_cmap(CMAP_NAME)

        plt.rcParams.update({'font.size': 14})
        for i,k in enumerate(experiment_constants.METRICS_K):
            metrics_mean=dict()

            for i,rec_using,metrics in zip(range(len(self.metrics)),self.metrics.keys(),self.metrics.values()):
                metrics=metrics[k]
                
                metrics_mean[rec_using]=defaultdict(lambda: defaultdict(float))
                for obj in metrics:
                    group = uid_group[obj['user_id']]
                    for key in self.metrics_name:
                        metrics_mean[rec_using][group][key]+=obj[key]
                
                for j,key in enumerate(self.metrics_name):
                    if key != 'epc':
                        for group in unique_groups:
                            metrics_mean[rec_using][group][key]/=groups_count[group]
                    else:
                        for group in unique_groups:
                            metrics_mean[rec_using][group][key] = self.groups_epc[rec_using][k][group]

                        
            reference_recommender = list(metrics_mean.keys())[0]
            reference_vals = metrics_mean.pop(reference_recommender)

            fig = plt.figure()
            ax=fig.add_subplot(111)
            num_recs = len(unique_groups)
            barWidth= 1-num_recs/(1+num_recs)
            N=len(self.metrics_name)
            indexes=np.arange(N)
            i=0
            styles = gen_bar_cycle(num_recs,ord_by_brightness=True,inverse_order=False)()
            rects = dict()
            for group in unique_groups:
                for rec_using,rec_metrics in metrics_mean.items():
                    print(group)
                    group_metrics = np.array(list(rec_metrics[group].values()))
                    # print(f"{rec_using} at @{k}")
                    # print(rec_metrics)
                    tmp_ref_val = np.array(list(reference_vals[group].values()))
                    # print(tmp_ref_val)
                    vals = 100*(group_metrics - tmp_ref_val)/tmp_ref_val
                    # vals= group_metrics
                    rects[group] = ax.bar(indexes+i*barWidth,vals,barWidth,label=rec_using,**next(styles))[0]
                    # for count, val in zip(indexes,vals):
                    #     ax.text(count+i*barWidth,val,"%s"%(group),rotation=90,zorder=33,fontweight='bold')
                    i+=1
            rects = list(rects.values())
            ax.set_xticks(np.arange(N+1)+barWidth*(((num_recs)/2)-1)+barWidth/2)
            if LANG == 'pt':
                tmp_legend_labels = list(map(lambda x:'G'+GROUP_ID.get(x),unique_groups))
            else:
                tmp_legend_labels = list(map(lambda x:'Group '+GROUP_ID.get(x),unique_groups))
            ax.legend(
                rects,
                tmp_legend_labels,handletextpad=-0.6,
                handler_map={rect: HandlerSquare() for rect in rects}
            )
            ax.set_xticklabels(map(lambda x: f'{x}@{k}',list(map(lambda name: METRICS_PRETTY[name],self.metrics_name))))

            if LANG == 'pt':
                ax.set_ylabel(f"Diferença relativa ao {reference_recommender}")
            else:
                ax.set_ylabel(f"Relative diff w.r.t. {reference_recommender}")
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())
            # ax.annotate(f'{group}',xy=(0,0.02),
            #                 xycoords='axes fraction')
            ax.set_xlim(-barWidth,len(self.metrics_name)-1+(num_recs-1)*barWidth+barWidth)
            for i in range(N):
                ax.axvline(i+num_recs*barWidth,color='k',linewidth=1,alpha=0.5)
            fig.savefig(self.data_directory+f"result/img/{prefix_name}_{self.base_rec}_{self.city}_{str(k)}.png",bbox_inches="tight")
            fig.savefig(self.data_directory+f"result/img/{prefix_name}_{self.base_rec}_{self.city}_{str(k)}.eps",bbox_inches="tight")

    def plot_recs_ild_gc_correlation(self,metrics=['ild','gc']):

        plt.rcParams.update({'font.size': 19})
        fig = plt.figure()
        ax = fig.add_subplot(111)

        colors = ord_scheme_brightness(MY_COLOR_SCHEME)[:len(self.metrics)]
        i=0
        for rec, metrics_ks in zip(self.metrics.keys(), self.metrics.values()):
            correlations = dict()
            for k, metrics_k in metrics_ks.items():
                print("%s@%d correlation"%(rec,k))
                df_p_metrics = pd.DataFrame(metrics_k)
                df_p_metrics = df_p_metrics.set_index('user_id')
                correlations[k] = abs(df_p_metrics[metrics].corr().loc['ild','gc'])
            ks = list(metrics_ks.keys())

            ax.plot(list(correlations.keys()),list(correlations.values()),marker='o',color=colors[i])
            i+=1
        ax.set_ylim(0,1)
        ax.set_xlabel(f'List size')
        ax.set_ylabel('Pearson correlation(Absolute)')
        xticks = np.round(np.linspace(ks[0],ks[-1],6))
        ax.set_xticks(xticks)
        ax.legend(list(self.metrics.keys()),frameon=False)
        fig.savefig(self.data_directory+IMG+f"correlation_{self.city}.png",bbox_inches="tight")
        fig.savefig(self.data_directory+IMG+f"correlation_{self.city}.eps",bbox_inches="tight")

    def print_latex_groups_cities_metrics_table(self,cities,prefix_name='',references=[],heuristic=False,print_htime=False):
        num_cities = len(cities)
        num_metrics = len(self.metrics_name)
        result_str = Text()
        result_str += r"\begin{table}[]\scriptsize"
        ls_str = ''.join([('l'*num_metrics+'|') if i!=(num_cities-1) else ('l'*num_metrics) for i in range(num_cities)])
        result_str += r"\begin{tabular}{" +'l|'+ls_str + "}"
        line_start = len(result_str)

        final_rec_list_size = self.final_rec_list_size

        for city in cities:
            self.city = city
            self.final_rec_list_size = final_rec_list_size
            # self.load_metrics(base=True,name_type=NameType.PRETTY)
            self.final_rec = 'persongeocat'
            self.final_rec_parameters = {'cat_div_method': 'inv_num_cat', 'geo_div_method': 'walk', 'norm_method': 'quadrant','bins': None,'funnel':None}
            uid_group = self.get_groups()

            groups_count = Counter(uid_group.values())
            unique_groups = list(groups_count.keys())
            unique_groups = [x for y, x in sorted(zip(iter(map(GROUP_ID.get,unique_groups)), unique_groups))]
        
            run_times = dict()
            
            if not heuristic:
                for rec in ['geocat','persongeocat']:
                    self.final_rec = rec
                    self.load_metrics(base=False,name_type=NameType.PRETTY)

            if not references:
                references = [list(self.metrics.keys())[0]]
            line_num = line_start
            if city != cities[-1]:
                result_str[line_num] += "\t&"+'\multicolumn{%d}{c|}{%s}' % (num_metrics,CITIES_PRETTY[city])
            else:
                result_str[line_num] += "\t&"+'\multicolumn{%d}{c}{%s}' % (num_metrics,CITIES_PRETTY[city])
            if city == cities[-1]:
                result_str[line_num] += '\\\\'
            line_num += 1
            for i,k in enumerate(experiment_constants.METRICS_K):

                if len(result_str[line_num]) == 0:
                    result_str[line_num] += "\\hline \\rowcolor{Gray} \\textbf{Algorithm} & "+'& '.join(map(lambda x: "\\textbf{"+METRICS_PRETTY[x]+f"@{k}}}" ,self.metrics_name))
                else:
                    result_str[line_num] += '& '+ '& '.join(map(lambda x: "\\textbf{"+METRICS_PRETTY[x]+f"@{k}}}" ,self.metrics_name))
                if city == cities[-1]:
                    result_str[line_num] += "\\\\"
                line_num += 1

                metrics_mean = dict()
                metrics_gain=dict()
                group_metrics = dict()
                for i,rec_using,metrics in zip(range(len(self.metrics)),self.metrics.keys(),self.metrics.values()):
                    metrics=metrics[k]

                    metrics_mean[rec_using]=defaultdict(lambda: defaultdict(float))
                    group_metrics[rec_using] = defaultdict(lambda: defaultdict(list))
                    for obj in metrics:
                        group = uid_group[obj['user_id']]
                        group_metrics[rec_using][k][group].append(obj)
                        for key in self.metrics_name:
                            metrics_mean[rec_using][group][key]+=obj[key]

                    for j,key in enumerate(self.metrics_name):
                        for group in unique_groups:
                            metrics_mean[rec_using][group][key]/=groups_count[group]
                            
                for i,rec_using,metrics in zip(range(len(self.metrics)),self.metrics.keys(),group_metrics.values()):
                    metrics_k=metrics[k]
                    metrics_gain[rec_using] = dict()
                    for group, group_metrics in metrics_k.items():
                        metrics_gain[rec_using][group] = dict()
                        if rec_using not in references:
                            for metric_name in self.metrics_name:
                                # print(group,len(group_metrics))
                                # print(group,len(base_metrics[k][group]))
                                statistic, pvalue = scipy.stats.wilcoxon(
                                    [ms[metric_name] for ms in base_metrics[k][group]],
                                    [ms[metric_name] for ms in group_metrics],
                                )
                                if pvalue > 0.05:
                                    metrics_gain[rec_using][group][metric_name] = bullet_str
                                else:
                                    if metrics_mean[rec_using][group][metric_name] < metrics_mean[reference_name][group][metric_name]:
                                        metrics_gain[rec_using][group][metric_name] = triangle_down_str
                                    elif metrics_mean[rec_using][group][metric_name] > metrics_mean[reference_name][group][metric_name]:
                                        metrics_gain[rec_using][group][metric_name] = triangle_up_str
                                    else:
                                        metrics_gain[rec_using][group][metric_name] = bullet_str 
                        else:
                            reference_name = rec_using
                            base_metrics = metrics
                            for metric_name in self.metrics_name:
                                metrics_gain[rec_using][group][metric_name] = ''
                                
                metrics_max = dict()
                for group in unique_groups:
                    metrics_max[group] = {mn:(None,0) for mn in self.metrics_name}
                for rec_using,rec_groups in metrics_mean.items():
                    for group, rec_metrics in rec_groups.items():
                        for metric_name, value in rec_metrics.items():
                            if metrics_max[group][metric_name][1] < value:
                                metrics_max[group][metric_name] = (rec_using,value)
                is_metric_the_max = defaultdict(lambda: defaultdict(lambda: defaultdict(bool)))
                for group in unique_groups:
                    for metric_name,(rec_using,value) in metrics_max[group].items():
                        is_metric_the_max[rec_using][group][metric_name] = True

                for group in unique_groups:
                    for rec_using,rec_metrics in metrics_mean.items():
                        rec_metrics=rec_metrics[group]
                        gain = metrics_gain[rec_using][group]

                        if city == cities[0]:
                            result_str[line_num] += rec_using + f"({group})"
                        result_str[line_num] += ' &' + '& '.join(map(lambda x: "\\textbf{%.4f}%s" %(x[0],x[1]) if is_metric_the_max[rec_using][group][x[2]] else "%.4f%s"%(x[0],x[1])  ,zip(rec_metrics.values(),gain.values(),rec_metrics.keys())))

                        if city == cities[-1]:
                            result_str[line_num] += "\\\\"
                        line_num += 1

        result_str += "\\end{tabular}"
        result_str += "\\end{table}"
        result_str = result_str.__str__()
        result_str = LATEX_HEADER+result_str
        result_str += LATEX_FOOT
        fout = open(self.data_directory+UTIL+'_'.join(references)+'_'+'side_'+'_'.join(([prefix_name] if len(prefix_name)>0 else [])+cities)+'.tex', 'w')
        fout.write(result_str)
        fout.close()

    def plot_precision_recall(self):
        rec = list(self.metrics.keys())[-1]
        metrics_ks = list(self.metrics.values())[-1]

        plt.rcParams.update({'font.size': 18})
        for k, metrics_k in metrics_ks.items():
            fig = plt.figure()
            ax=fig.add_subplot(111)
            colors = get_my_color_scheme(2,ord_by_brightness=True,inverse_order=False)
            df_p_metrics = pd.DataFrame(metrics_k)
            df_p_metrics = df_p_metrics.set_index('user_id').sort_values(by=['precision','recall'],ascending=False)
            ax.plot(df_p_metrics['recall'],df_p_metrics['precision'],color='k')
            ax.set_xlabel(f"Recall@{k}")
            ax.set_ylabel(f"Precision@{k}")
            ax.set_title(f"{rec} in {CITIES_PRETTY[self.city]}")
            print(df_p_metrics[['recall','precision']])
            fig.savefig(self.data_directory+IMG+f"prec_rec_{self.city}_{rec}_{k}.png",bbox_inches="tight")
        
    def print_groups_info(self):
        df = pd.DataFrame([],
                          columns=[
                              'visits','visits_mean',
                              'visits_std','cats_visited',
                              'cats_visited_mean','cats_visited_std',
                              'num_friends'
                          ]
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
            num_friends = len(self.social_relations[uid])
            df.loc[uid] = [visits,visits_mean,
                           visits_std,len(cats_visits),
                           cats_visits_mean,cats_visits_std,num_friends] + methods_values
        
        self.final_rec = 'persongeocat'
        self.final_rec_parameters = {'cat_div_method': 'inv_num_cat', 'geo_div_method': 'walk', 'norm_method': 'quadrant','bins': None,'funnel':None}
        uid_group = self.get_groups()

        groups_count = Counter(uid_group.values())
        unique_groups = list(groups_count.keys())
        unique_groups = [x for y, x in sorted(zip(iter(map(GROUP_ID.get,unique_groups)), unique_groups))]

        for group in unique_groups:
            uids = [uid for uid, gid in uid_group.items() if gid == group]
            print(group)
            print(df.loc[uids].describe())
        pass


    def plot_recs_precision_recall(self, METRICS_KS = experiment_constants.METRICS_K, metrics=['recall','precision']):
        # rec = list(self.metrics.keys())
        # metrics_ks = list(self.metrics.values())

        plt.rcParams.update({'font.size': 18})
        for i, k in enumerate(METRICS_KS):
            for metric_name in metrics:
                fig = plt.figure()
                ax=fig.add_subplot(111)
                colors = get_my_color_scheme(len(self.metrics),ord_by_brightness=True,inverse_order=False)
                j = 0
                # min_index_not_zero = math.inf
                for rec, metrics_ks in self.metrics.items():
                    metrics_k = metrics_ks[k]
                    user_num = len(metrics_k)
                    df_p_metrics = pd.DataFrame(metrics_k)
                    # if j == 0:
                    #     sorted_idxs = df_p_metrics[metric_name].argsort()

                    # df_p_metrics = df_p_metrics.iloc[sorted_idxs]
                    # df_p_metrics = df_p_metrics.sort_values(by=[metric_name])
                    # df_p_metrics = df_p_metrics.reset_index()
                    # not_zero_idx = (df_p_metrics[metric_name] > 0).idxmax()
                    # if not_zero_idx < min_index_not_zero:
                    #     min_index_not_zero = not_zero_idx
                    val = df_p_metrics[metric_name].to_numpy()
                    print(val)
                    xs = np.linspace(0,1,21)
                    ys = []
                    for x in xs:
                        ys.append(len(np.nonzero(val<=x)[0])/user_num)
                    # df_p_metrics = df_p_metrics.sort_values(by=[metric_name],ascending=False)
                    ax.plot(xs,ys,colors[j])
                    j+=1
                # ax.set_xlim(min_index_not_zero,user_num)
                ax.legend(list(self.metrics.keys()))
                # ax.set_xlabel(f"Recall@{k}")
                ax.set_xlabel("x")
                ax.set_ylabel(f"$P({metric_name}@{k}\leq x)$")
                # ax.set_ylabel(f"Precision@{k}")
                ax.set_title(f"{CITIES_PRETTY[self.city]}")
                fig.savefig(self.data_directory+IMG+f"prec_rec_{self.city}_{metric_name}_{k}.png",bbox_inches="tight")

    def print_discover_mean_cities(self):
        cities = ['lasvegas','phoenix']
        final_rec = 'geocat'

        num = len(cities)*len(experiment_constants.METRICS_K)
        cities_metrics_mean = np.zeros(len(self.metrics_name))
        for city in cities:
            self.city = city
            self.load_metrics(base=True,name_type=NameType.PRETTY)
            self.load_metrics(base=False,name_type=NameType.PRETTY)
           

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
                reference_recommender = list(metrics_mean.keys())[0]
                reference_vals = metrics_mean.pop(reference_recommender)

                for rec_using,rec_metrics in metrics_mean.items():
                    cities_metrics_mean += -100+100*np.array(list(rec_metrics.values()))/np.array(list(reference_vals.values()))
        print(f"MEAN GAIN in {', '.join(cities)}")
        print({metric_name: val for metric_name, val in zip(self.metrics_name,cities_metrics_mean/num)})
        
        pass


    def print_latex_vert_cities_metrics_table(self,cities,prefix_name='',references=[],heuristic=False,recs=['gc','ld','binomial','pm2','geodiv','geocat']):
        num_metrics = len(self.metrics_name)
        if heuristic:
            num_metrics += 1
        result_str = r"\begin{table}[]" + "\n"
        result_str += r"\begin{tabular}{" +'l|'+'l'*(num_metrics) + "}\n"
        # result_str += "\begin{tabular}{" + 'l'*(num_metrics+1) + "}\n"

        if cities == None:
            cities = [self.city]
        final_rec_list_size = self.final_rec_list_size

        for city in cities:
            self.city = city
            self.final_rec_list_size = final_rec_list_size

            self.load_metrics(base=True,name_type=NameType.PRETTY)

            run_times = dict()

            if not references:
                references = [list(self.metrics.keys())[0]]

            if not heuristic:
                for rec in recs:
                    self.final_rec = rec
                    self.load_metrics(base=False,name_type=NameType.PRETTY)

            result_str += "\t&"+'\multicolumn{%d}{c}{%s}\\\\\n' % (num_metrics,CITIES_PRETTY[city])

            if heuristic:
                self.final_rec_parameters = {'heuristic': 'local_max'}
                self.load_metrics(base=False,name_type=NameType.PRETTY)

            for i,k in enumerate(experiment_constants.METRICS_K):

                if heuristic:
                    self.final_rec_list_size = k
                    for h in ['tabu_search', 'particle_swarm']:
                        self.final_rec_parameters = {'heuristic': h}
                        self.load_metrics(base=False,name_type=NameType.PRETTY,METRICS_KS=[k])
                        run_times[self.get_final_rec_pretty_name()]=int(float(open(self.data_directory+UTIL+f'run_time_{self.get_final_rec_name()}.txt',"r").read()))
                    self.final_rec_parameters = {'heuristic': 'local_max'}
                    run_times[self.get_final_rec_pretty_name()]=int(float(open(self.data_directory+UTIL+f'run_time_{self.get_final_rec_name()}.txt',"r").read()))
                    max_arg_run_time = max(run_times, key=run_times.get)

                result_str += "\\hline \\rowcolor{Gray} \\textbf{Algorithm} & "+'& '.join(map(lambda x: "\\textbf{"+METRICS_PRETTY[x]+f"@{k}}}" ,self.metrics_name))

                if heuristic:
                    result_str += '& \\textbf{Time}'
                result_str += "\\\\\n"
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
                                metrics_gain[rec_using][metric_name] = bullet_str
                            else:
                                if metrics_mean[rec_using][metric_name] < metrics_mean[reference_name][metric_name]:
                                    metrics_gain[rec_using][metric_name] = triangle_down_str
                                elif metrics_mean[rec_using][metric_name] > metrics_mean[reference_name][metric_name]:
                                    metrics_gain[rec_using][metric_name] = triangle_up_str
                                else:
                                    metrics_gain[rec_using][metric_name] = bullet_str 
                    else:
                        reference_name = rec_using
                        base_metrics = metrics
                        metrics_gain[rec_using] = dict()
                        for metric_name in self.metrics_name:
                            metrics_gain[rec_using][metric_name] = ''

                metrics_max = {mn:(None,0) for mn in self.metrics_name}
                for rec_using,rec_metrics in metrics_mean.items():
                    for metric_name, value in rec_metrics.items():
                        if metrics_max[metric_name][1] < value:
                            metrics_max[metric_name] = (rec_using,value)
                is_metric_the_max = defaultdict(lambda: defaultdict(bool))
                for metric_name,(rec_using,value) in metrics_max.items():
                    is_metric_the_max[rec_using][metric_name] = True

                for rec_using,rec_metrics in metrics_mean.items():
                    gain = metrics_gain[rec_using]
                    result_str += rec_using + ' &' + '& '.join(map(lambda x: "\\textbf{%.4f}%s" %(x[0],x[1]) if is_metric_the_max[rec_using][x[2]] else "%.4f%s"%(x[0],x[1])  ,zip(rec_metrics.values(),gain.values(),rec_metrics.keys())))
                    if heuristic:
                        if rec_using not in references:
                            if rec_using != max_arg_run_time:
                                result_str += f'& {sec_to_hm(run_times[rec_using])}'
                            else:
                                result_str += f'& \\textbf{{{sec_to_hm(run_times[rec_using])}}}'
                        else:
                            result_str += f'& '
                    result_str += "\\\\\n"
            if city != cities[-1]:
                result_str += '\\hline'


        result_str += "\\end{tabular}\n"
        result_str += "\\end{table}\n"
        result_str = LATEX_HEADER + result_str
        result_str += LATEX_FOOT
        fout = open(self.data_directory+UTIL+'_'.join(([prefix_name] if len(prefix_name)>0 else [])+cities)+'.tex', 'w')
        fout.write(result_str)
        fout.close()


    def print_latex_discover_cities_metrics_table(self,cities,prefix_name='gc_ld',references=[],heuristic=False,phis=[0.0,1.0],METRICS_KS=[10]):
        num_metrics = len(self.metrics_name)
        if heuristic:
            num_metrics += 1
        result_str = r"\begin{table}[]" + "\n"
        result_str += r"\begin{tabular}{" +'l|'+'l'*(num_metrics) + "}\n"
        # result_str += "\begin{tabular}{" + 'l'*(num_metrics+1) + "}\n"

        if cities == None:
            cities = [self.city]
        final_rec_list_size = self.final_rec_list_size

        for city in cities:
            self.city = city
            self.final_rec_list_size = final_rec_list_size
            self.final_rec_parameters = {}
            self.load_metrics(base=False,name_type=NameType.PRETTY)

            run_times = dict()

            if not references:
                references = [list(self.metrics.keys())[0]]

            if not heuristic:
                for phi in phis:
                    self.final_rec_parameters = {'div_geo_cat_weight':1.0,'div_cat_weight':phi}
                    self.load_metrics(base=False,name_type=NameType.PRETTY)

            result_str += "\t&"+'\multicolumn{%d}{c}{%s}\\\\\n' % (num_metrics,CITIES_PRETTY[city])

            if heuristic:
                self.final_rec_parameters = {'heuristic': 'local_max'}
                self.load_metrics(base=False,name_type=NameType.PRETTY)

            for i,k in enumerate(METRICS_KS):

                if heuristic:
                    self.final_rec_list_size = k
                    for h in ['tabu_search', 'particle_swarm']:
                        self.final_rec_parameters = {'heuristic': h}
                        self.load_metrics(base=False,name_type=NameType.PRETTY,METRICS_KS=[k])
                        run_times[self.get_final_rec_pretty_name()]=int(float(open(self.data_directory+UTIL+f'run_time_{self.get_final_rec_name()}.txt',"r").read()))
                    self.final_rec_parameters = {'heuristic': 'local_max'}
                    run_times[self.get_final_rec_pretty_name()]=int(float(open(self.data_directory+UTIL+f'run_time_{self.get_final_rec_name()}.txt',"r").read()))
                    max_arg_run_time = max(run_times, key=run_times.get)

                result_str += "\\hline \\rowcolor{Gray} \\textbf{Algorithm} & "+'& '.join(map(lambda x: "\\textbf{"+METRICS_PRETTY[x]+f"@{k}}}" ,self.metrics_name))

                if heuristic:
                    result_str += '& \\textbf{Time}'
                result_str += "\\\\\n"
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
                                metrics_gain[rec_using][metric_name] = bullet_str
                            else:
                                if metrics_mean[rec_using][metric_name] < metrics_mean[reference_name][metric_name]:
                                    metrics_gain[rec_using][metric_name] = triangle_down_str
                                elif metrics_mean[rec_using][metric_name] > metrics_mean[reference_name][metric_name]:
                                    metrics_gain[rec_using][metric_name] = triangle_up_str
                                else:
                                    metrics_gain[rec_using][metric_name] = bullet_str 
                    else:
                        reference_name = rec_using
                        base_metrics = metrics
                        metrics_gain[rec_using] = dict()
                        for metric_name in self.metrics_name:
                            metrics_gain[rec_using][metric_name] = ''

                metrics_max = {mn:(None,0) for mn in self.metrics_name}
                for rec_using,rec_metrics in metrics_mean.items():
                    for metric_name, value in rec_metrics.items():
                        if metrics_max[metric_name][1] < value:
                            metrics_max[metric_name] = (rec_using,value)
                is_metric_the_max = defaultdict(lambda: defaultdict(bool))
                for metric_name,(rec_using,value) in metrics_max.items():
                    is_metric_the_max[rec_using][metric_name] = True

                for rec_using,rec_metrics in metrics_mean.items():
                    gain = metrics_gain[rec_using]
                    result_str += rec_using + ' &' + '& '.join(map(lambda x: "\\textbf{%.4f}%s" %(x[0],x[1]) if is_metric_the_max[rec_using][x[2]] else "%.4f%s"%(x[0],x[1])  ,zip(rec_metrics.values(),gain.values(),rec_metrics.keys())))
                    if heuristic:
                        if rec_using not in references:
                            if rec_using != max_arg_run_time:
                                result_str += f'& {sec_to_hm(run_times[rec_using])}'
                            else:
                                result_str += f'& \\textbf{{{sec_to_hm(run_times[rec_using])}}}'
                        else:
                            result_str += f'& '
                    result_str += "\\\\\n"
            if city != cities[-1]:
                result_str += '\\hline'


        result_str += "\\end{tabular}\n"
        result_str += "\\end{table}\n"
        result_str = LATEX_HEADER + result_str
        result_str += LATEX_FOOT
        fout = open(self.data_directory+UTIL+'_'.join(([prefix_name] if len(prefix_name)>0 else [])+cities)+'.tex', 'w')
        fout.write(result_str)
        fout.close()


    def print_binomial_hyperparameter(self,inl):
        KS = [10]
        
        l = []
        for i in inl:
            for j in inl:
                if not(i == 0 and j != 0):
                    l.append((i,j))
                    self.final_rec_parameters['div_weight'], self.final_rec_parameters['alpha'] = i, j
                    self.load_metrics(base=False,name_type=NameType.FULL,METRICS_KS=KS)
        for i,k in enumerate(KS):
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

            print(pd.Series(mauts).sort_values(ascending=False))

    def geomf(self):
        geomf = GeoMF(**self.base_rec_parameters)
        geomf.train(self.training_matrix,self.poi_coos)
        self.cache['geomf'] = geomf
        args=[(uid,) for uid in self.all_uids]
        results = run_parallel(self.run_geomf,args,self.CHKS)
        self.save_result(results,base=True)

    @classmethod
    def run_geomf(cls, uid):
        self = cls.getInstance()
        geomf = self.cache['geomf']
        if uid in self.ground_truth:

            geomf_scores = geomf.predict(uid,self.all_lids)[0]
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
        geodiv2020.train(self.training_matrix,self.poi_coos)
        # print("exiting!!")
        # raise SystemExit
        self.cache['geodiv2020'] = geodiv2020
        args=[(uid,) for uid in self.all_uids]
        results = run_parallel(self.run_geodiv2020,args,self.CHKS)
        self.save_result(results,base=False)

    @classmethod
    def run_geodiv2020(cls, uid):
        self = cls.getInstance()
        geodiv2020 = self.cache['geodiv2020']
        if uid in self.ground_truth:
            predicted = self.user_base_predicted_lid[uid][
                0:self.base_rec_list_size]
            overall_scores = self.user_base_predicted_score[uid][
                0:self.base_rec_list_size]
            
            predicted, overall_scores = geodiv2020.predict(uid,predicted,overall_scores,self.final_rec_list_size)
            # print(np.min(overall_scores),np.max(overall_scores))
            # print(predicted)
            # print(predicted,overall_scores)
            assert(self.final_rec_list_size == len(predicted))

            return json.dumps({'user_id': uid, 'predicted': list(map(int, predicted)), 'score': list(map(float, overall_scores))})+"\n"
        self.not_in_ground_truth_message(uid)
        return ""
