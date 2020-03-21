class experiment_constants:
    CITIES=['madison','charlotte','lasvegas','phoenix','montreal','pittsburgh']
    CITY="madison" # city source of checkins
    TRAIN_SIZE=0.7
    N=80
    K=20
    METRICS_K=[5,10,20]
class geocat_constants:
    NEIGHBOR_DISTANCE = 0.5#km
    N = 80# temp list size
    K = 20# final list size
    VERY_SMALL_VALUE = -100 # used for objective function
    DIV_GEO_CAT_WEIGHT = 0.5 # beta,this is here because of the work to be done on parameter customization for each user
    DIV_WEIGHT=0.75 # lambda, geo vs cat_DIV_WEIGHT = 0.75 # lambda, geo vs cat


class usg_constants:
    eta=0.05


METRICS_PRETTY = {'precision':'Prec',
                  'recall':'Rec',
                  'ild': 'ILD',
                  'gc': 'GC',
                  'pr':'PRg',
                  'epc':'EPC',
                  'ndcg':'NDCG',
                  'map':'MAP',
                  'ildg':'ILDg',
                  'maut':'MAUT',
                  'f1': 'F1',
}

RECS_PRETTY = {
    "usg": "USG",
    "mostpopular": "MostPopular",
    "geocat": "DisCovER",
    "persongeocat": "PersonDisCovER",
    "geodiv": "Geo-Div(PR)",
    "ld": "LD",
    "binomial": "Binom",
    "pm2": "PM2",
    "perfectpgeocat": "PPGC",
    "pdpgeocat": "PDPGC",
    "geosoca": "GeoSoCa",
    "gc": "GC",
}


CITIES_PRETTY = {
    'madison':'Madison',
    'charlotte':'Charlotte',
    'lasvegas':'Las Vegas',
    'phoenix':'Phoenix',
    'montreal':'Montreal',
    'pittsburgh':'Pittsburgh'
}

HEURISTICS_PRETTY = {
    'local_max': 'GRD',
    'particle_swarm': 'PSO',
    'tabu_search': 'TS',
}

SIZE_AWARENESS_RECS = { # unused
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

GROUP_ID = {
    'geo_preference': '1',
    'geocat_preference': '2',
    'no_preference': '3',
    'cat_preference': '4',
}
