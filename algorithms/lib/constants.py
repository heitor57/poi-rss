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
}

RECS_PRETTY = {
    "usg": "USG",
    "mostpopular": "MostPopular",
    "geocat": "Geo-Cat",
    "pgeocat": "PersonGeoCat",
    "geodiv": "Geo-Div",
    "ld": "LD",
    "binomial": "Binom",
    "pm2": "PM2",
    "perfectpgeocat": "PPGC",
    "pdpgeocat": "PDPGC",
    "geosoca": "GeoSoCa",
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
    'local_max': 'LM',
    'particle_swarm': 'PSO',
    'tabu_search': 'TS',
}
