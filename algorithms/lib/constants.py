class experiment_constants:
    CITY="madison" # city source of checkins

class geocat_constants:
    NEIGHBOR_DISTANCE = 0.5#km
    N = 80# temp list size
    K = 20# final list size
    VERY_SMALL_VALUE = -100 # used for objective function
    DIV_GEO_CAT_WEIGHT = 0.5 # beta,this is here because of the work to be done on parameter customization for each user
    DIV_WEIGHT=0.75 # lambda, geo vs cat_DIV_WEIGHT = 0.75 # lambda, geo vs cat


class usg_constants:
    eta=0.05