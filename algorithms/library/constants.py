class experiment_constants:
    CITIES = ['madison', 'charlotte', 'lasvegas',
              'phoenix', 'montreal', 'pittsburgh']
    CITY = "madison"  # city source of checkins
    TRAIN_SIZE = 0.7
    N = 80
    K = 20
    METRICS_K = [5, 10, 20]


R_FORMAT = '.json'  # Results format is json, metrics, rec lists, etc
D_FORMAT = '.pickle'  # data set format is pickle

DATA = '../../data'
DATASET_DIRECTORY = 'datasets/'
RESULT_DIRECTORY = 'results/'
TRAIN = DATASET_DIRECTORY+'checkin/train/'  # train data sets
TEST = DATASET_DIRECTORY+'checkin/test/'  # test data sets
POI = DATASET_DIRECTORY+'poi/'  # poi data sets with cats and coos
POI_FULL = DATASET_DIRECTORY+'poi_full/'  # poi data sets with cats full without preprocessing
USER_FRIEND = DATASET_DIRECTORY+'user/friend/'  # users and friends
USER = DATASET_DIRECTORY+'user/'  # user general data
NEIGHBOR = DATASET_DIRECTORY+'neighbor/'  # neighbors of pois

METRICS = RESULT_DIRECTORY+'metrics/'
RECLIST = RESULT_DIRECTORY+'reclist/'
IMG = RESULT_DIRECTORY+'img/'
UTIL = RESULT_DIRECTORY+'util/'

# class geocat_constants:
    # NEIGHBOR_DISTANCE = 0.5  # km
    # N = 80  # temp list size
    # K = 20  # final list size
    # VERY_SMALL_VALUE = -100  # used for objective function
    # # beta,this is here because of the work to be done on parameter customization for each user
    # DIV_GEO_CAT_WEIGHT = 0.5
    # DIV_WEIGHT = 0.75  # lambda, geo vs cat_DIV_WEIGHT = 0.75 # lambda, geo vs cat


EARTH_RADIUS = 6371


# class usg_constants:
    # eta = 0.05


METRICS_PRETTY = {'precision': 'Prec',
                  'recall': 'Rec',
                  'ild': 'ILD',
                  'gc': 'Cov',
                  'pr': 'PRg',
                  'epc': 'EPC',
                  'ndcg': 'NDCG',
                  'map': 'MAP',
                  'ildg': 'ILDg',
                  'maut': 'MAUT',
                  'f1': 'F1',
                  }

RECS_PRETTY = {
    "usg": "USG",
    "mostpopular": "MostPopular",
    "geomf": "GeoMF",
    # "geocat": "DisCovER",
    "geocat": "DisCovER",
    "persongeocat": "PersonDisCovER",
    # "geodiv": "Geo-Div(PR)",
    "geodiv": "GeoDiv",
    "ld": "LD",
    "binomial": "Binom",
    "pm2": "PM2",
    "perfectpgeocat": "PPGC",
    "pdpgeocat": "PDPGC",
    "geosoca": "GeoSoCa",
    "gc": "GC",
    "geodiv2020": "Geo-Div",
}


CITIES_PRETTY = {
    'madison': 'Madison',
    'charlotte': 'Charlotte',
    'lasvegas': 'Las Vegas',
    'phoenix': 'Phoenix',
    'montreal': 'Montreal',
    'pittsburgh': 'Pittsburgh'
}

HEURISTICS_PRETTY = {
    'local_max': 'GRD',
    'particle_swarm': 'PSO',
    'tabu_search': 'TS',
}

SIZE_AWARENESS_RECS = {  # unused
    "usg": False,
    "mostpopular": False,
    "geocat": False,
    "persongeocat": False,
    "geodiv": False,
    "ld": False,
    "binomial": True,
    "pm2": False,
    "perfectpgeocat": False,
    "pdpgeocat": False,
    "geosoca": False,
}

# GROUP_ID = {
    # 'geo_preference': '1',
    # 'geocat_preference': '2',
    # 'no_preference': '3',
    # 'cat_preference': '4',
# }
# USG

# CITIES_BEST_PARAMETERS = {
#     'lasvegas': {
#         "geocat": {'div_weight':0.75,'div_geo_cat_weight':0.25, 'heuristic': 'local_max', 'obj_func': 'cat_weight', 'div_cat_weight': 0.05},
#         "geodiv": {'div_weight':0.5},
#         "ld": {'div_weight':0.25},
#         "binomial": {'alpha': 0.5, 'div_weight': 0.75},
#         "pm2": {'div_weight': 1},
#         "gc": {'div_weight': 0.8},
#         "persongeocat": {'div_weight':1.0,'cat_div_method': 'inv_num_cat',
#                          'geo_div_method': 'walk', 'obj_func': 'cat_weight',
#                          'div_cat_weight':0.05, 'bins': None,
#                          'norm_method': 'default','funnel':None},
#         "geodiv2020": {'div_weight': 0.5},
#     },
#     'phoenix': {
#         "geocat": {'div_weight':0.75,'div_geo_cat_weight':0.25, 'heuristic': 'local_max', 'obj_func': 'cat_weight', 'div_cat_weight': 0.05},
#         "geodiv": {'div_weight':0.5},
#         "ld": {'div_weight':0.25},
#         "binomial": {'alpha': 0.5, 'div_weight': 0.75},
#         "pm2": {'div_weight': 1},
#         "gc": {'div_weight': 0.8},
#         "persongeocat": {'div_weight':1.0,'cat_div_method': 'inv_num_cat',
#                          'geo_div_method': 'walk', 'obj_func': 'cat_weight',
#                          'div_cat_weight':0.05, 'bins': None,
#                          'norm_method': 'default','funnel':None},
#         "geodiv2020": {'div_weight': 0.25},
#     },
# }

# GEOSO
CITIES_BEST_PARAMETERS = {
    'geosoca': {
        'madison': {  # copied from lasvegas
            "geocat": {'div_weight': 0.75, 'div_geo_cat_weight': 0.25, 'heuristic': 'local_max', 'obj_func': 'cat_weight', 'div_cat_weight': 0.05},
            "geodiv": {'div_weight': 0.5},
            "ld": {'div_weight': 0.25},
            "binomial": {'alpha': 0.5, 'div_weight': 0.75},
            "pm2": {'div_weight': 1},
            "gc": {'div_weight': 0.8},
            "persongeocat": {'div_weight': 1.0, 'cat_div_method': 'inv_num_cat',
                             'geo_div_method': 'walk', 'obj_func': 'cat_weight',
                             'div_cat_weight': 0.05, 'bins': None,
                             'norm_method': 'default', 'funnel': None},
            "geodiv2020": {'div_weight': 0.25},
        },
        'lasvegas': {
            "geocat": {'div_weight': 0.75, 'div_geo_cat_weight': 0.25, 'heuristic': 'local_max', 'obj_func': 'cat_weight', 'div_cat_weight': 0.05},
            "geodiv": {'div_weight': 0.5},
            "ld": {'div_weight': 0.25},
            "binomial": {'alpha': 0.5, 'div_weight': 0.75},
            "pm2": {'div_weight': 1},
            "gc": {'div_weight': 0.8},
            "persongeocat": {'div_weight': 1.0, 'cat_div_method': 'inv_num_cat',
                             'geo_div_method': 'walk', 'obj_func': 'cat_weight',
                             'div_cat_weight': 0.05, 'bins': None,
                             'norm_method': 'default', 'funnel': None},
            "geodiv2020": {'div_weight': 0.25},
        },
        'phoenix': {
            "geocat": {'div_weight': 0.75, 'div_geo_cat_weight': 0.25, 'heuristic': 'local_max', 'obj_func': 'cat_weight', 'div_cat_weight': 0.05},
            "geodiv": {'div_weight': 0.5},
            "ld": {'div_weight': 0.25},
            "binomial": {'alpha': 0.5, 'div_weight': 0.75},
            "pm2": {'div_weight': 1},
            "gc": {'div_weight': 0.8},
            "persongeocat": {'div_weight': 1.0, 'cat_div_method': 'inv_num_cat',
                             'geo_div_method': 'walk', 'obj_func': 'cat_weight',
                             'div_cat_weight': 0.05, 'bins': None,
                             'norm_method': 'default', 'funnel': None},
            "geodiv2020": {'div_weight': 0.25},
        }, },
    'usg': {
        'charlotte': {  # copied from geosoca lasvegas
            "geocat": {'div_weight': 0.75, 'div_geo_cat_weight': 0.25, 'heuristic': 'local_max', 'obj_func': 'cat_weight', 'div_cat_weight': 0.05},
            "geodiv": {'div_weight': 0.5},
            "ld": {'div_weight': 0.25},
            "binomial": {'alpha': 0.5, 'div_weight': 0.75},
            "pm2": {'div_weight': 1},
            "gc": {'div_weight': 0.8},
            "persongeocat": {'div_weight': 1.0, 'cat_div_method': 'inv_num_cat',
                             'geo_div_method': 'walk', 'obj_func': 'cat_weight',
                             'div_cat_weight': 0.05, 'bins': None,
                             'norm_method': 'default', 'funnel': None},
            "geodiv2020": {'div_weight': 0.25},
        },
        'madison': {  # copied from geosoca lasvegas
            "geocat": {'div_weight': 0.75, 'div_geo_cat_weight': 0.25, 'heuristic': 'local_max', 'obj_func': 'cat_weight', 'div_cat_weight': 0.05},
            "geodiv": {'div_weight': 0.5},
            "ld": {'div_weight': 0.25},
            "binomial": {'alpha': 0.5, 'div_weight': 0.75},
            "pm2": {'div_weight': 1},
            "gc": {'div_weight': 0.8},
            "persongeocat": {'div_weight': 1.0, 'cat_div_method': 'inv_num_cat',
                             'geo_div_method': 'walk', 'obj_func': 'cat_weight',
                             'div_cat_weight': 0.05, 'bins': None,
                             'norm_method': 'default', 'funnel': None},
            "geodiv2020": {'div_weight': 0.25},
        },
        'lasvegas': {
            "geocat": {'div_weight': 0.75, 'div_geo_cat_weight': 0.25, 'heuristic': 'local_max', 'obj_func': 'cat_weight', 'div_cat_weight': 0.05},
            "geodiv": {'div_weight': 0.5},
            "ld": {'div_weight': 0.25},
            "binomial": {'alpha': 0.5, 'div_weight': 0.75},
            "pm2": {'div_weight': 1},
            "gc": {'div_weight': 0.8},
            "persongeocat": {'div_weight': 1.0, 'cat_div_method': 'inv_num_cat',
                             'geo_div_method': 'walk', 'obj_func': 'cat_weight',
                             'div_cat_weight': 0.05, 'bins': None,
                             'norm_method': 'default', 'funnel': None},
            "geodiv2020": {'div_weight': 0.5},
        },
        'phoenix': {
            "geocat": {'div_weight': 0.75, 'div_geo_cat_weight': 0.25, 'heuristic': 'local_max', 'obj_func': 'cat_weight', 'div_cat_weight': 0.05},
            "geodiv": {'div_weight': 0.5},
            "ld": {'div_weight': 0.25},
            "binomial": {'alpha': 0.5, 'div_weight': 0.75},
            "pm2": {'div_weight': 1},
            "gc": {'div_weight': 0.8},
            "persongeocat": {'div_weight': 1.0, 'cat_div_method': 'inv_num_cat',
                             'geo_div_method': 'walk', 'obj_func': 'cat_weight',
                             'div_cat_weight': 0.05, 'bins': None,
                             'norm_method': 'default', 'funnel': None},
            "geodiv2020": {'div_weight': 0.25},
        }, },
    'geomf': {
        'madison': {  # copied from geosoca lasvegas
            "geocat": {'div_weight': 0.75, 'div_geo_cat_weight': 0.25, 'heuristic': 'local_max', 'obj_func': 'cat_weight', 'div_cat_weight': 0.05},
            "geodiv": {'div_weight': 0.5},
            "ld": {'div_weight': 0.25},
            "binomial": {'alpha': 0.5, 'div_weight': 0.75},
            "pm2": {'div_weight': 1},
            "gc": {'div_weight': 0.8},
            "persongeocat": {'div_weight': 1.0, 'cat_div_method': 'inv_num_cat',
                             'geo_div_method': 'walk', 'obj_func': 'cat_weight',
                             'div_cat_weight': 0.05, 'bins': None,
                             'norm_method': 'default', 'funnel': None},
            "geodiv2020": {'div_weight': 0.25},
        },
        'lasvegas': {
            "geocat": {'div_weight': 0.75, 'div_geo_cat_weight': 0.25, 'heuristic': 'local_max', 'obj_func': 'cat_weight', 'div_cat_weight': 0.05},
            "geodiv": {'div_weight': 0.1},
            "ld": {'div_weight': 0.1},
            "binomial": {'alpha': 0.5, 'div_weight': 0.75},
            "pm2": {'div_weight': 0.9},
            "gc": {'div_weight': 0.6},
            "geodiv2020": {'div_weight': 0.5},
            # "persongeocat": {'div_weight':1.0,'cat_div_method': 'inv_num_cat',
            #                  'geo_div_method': 'walk', 'obj_func': 'cat_weight',
            #                  'div_cat_weight':0.05, 'bins': None,
            #                  'norm_method': 'default','funnel':None},
        },
        'phoenix': {
            "geocat": {'div_weight': 0.75, 'div_geo_cat_weight': 0.25, 'heuristic': 'local_max', 'obj_func': 'cat_weight', 'div_cat_weight': 0.05},
            "ld": {'div_weight': 0.1},
            "gc": {'div_weight': 0.7},
            "pm2": {'div_weight': 0.9},
            "geodiv": {'div_weight': 0.1},
            "binomial": {'alpha': 1.0, 'div_weight': 1.0},
            "geodiv2020": {'div_weight': 0.25},
        }, },
}
# GS
# CITIES_BEST_PARAMETERS = {
#     'lasvegas': {
#         "geocat": {'div_weight':1.0,'div_geo_cat_weight':0.25, 'heuristic': 'local_max', 'obj_func': 'cat_weight', 'div_cat_weight': 0.05},
#         "geodiv": {'div_weight':0.5},
#         "ld": {'div_weight':0.25},
#         "binomial": {'alpha': 1.0, 'div_weight': 1.0},
#         "pm2": {'div_weight': 0.6},
#         "gc": {'div_weight': 0.7},
#         "persongeocat": {'div_weight':1.0,'cat_div_method': 'inv_num_cat',
#                          'geo_div_method': 'walk', 'obj_func': 'cat_weight',
#                          'div_cat_weight':0.05, 'bins': None,
#                          'norm_method': 'default','funnel':None},
#         "geodiv2020": {'div_weight': 0.25},
#     },
#     'phoenix': {
#             "geocat": {'div_weight':1.0,'div_geo_cat_weight':0.25, 'heuristic': 'local_max', 'obj_func': 'cat_weight', 'div_cat_weight': 0.05},
#             "ld": {'div_weight':0.4},
#             "gc": {'div_weight':0.3},
#             "pm2": {'div_weight':1.0},
#             "geodiv": {'div_weight':1.0},
#             "binomial": {'alpha': 1.0, 'div_weight': 1.0},
#         "geodiv2020": {'div_weight': 0.25},
#     },

# }


# GEOMF
# CITIES_BEST_PARAMETERS = {
#     'lasvegas': {
#         "geocat": {'div_weight':0.75,'div_geo_cat_weight':0.25, 'heuristic': 'local_max', 'obj_func': 'cat_weight', 'div_cat_weight': 0.05},
#         "geodiv": {'div_weight':0.1},
#         "ld": {'div_weight':0.1},
#         "binomial": {'alpha': 0.5, 'div_weight': 0.75},
#         "pm2": {'div_weight': 0.9},
#         "gc": {'div_weight': 0.6},
#         "geodiv2020": {'div_weight': 0.5},
#         # "persongeocat": {'div_weight':1.0,'cat_div_method': 'inv_num_cat',
#         #                  'geo_div_method': 'walk', 'obj_func': 'cat_weight',
#         #                  'div_cat_weight':0.05, 'bins': None,
#         #                  'norm_method': 'default','funnel':None},
#     },
#     'phoenix': {
#             "geocat": {'div_weight':0.75,'div_geo_cat_weight':0.25, 'heuristic': 'local_max', 'obj_func': 'cat_weight', 'div_cat_weight': 0.05},
#             "ld": {'div_weight':0.1},
#             "gc": {'div_weight':0.7},
#             "pm2": {'div_weight':0.9},
#             "geodiv": {'div_weight':0.1},
#             "binomial": {'alpha': 1.0, 'div_weight': 1.0},
#         "geodiv2020": {'div_weight': 0.25},
#     },

# }
