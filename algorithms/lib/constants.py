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
