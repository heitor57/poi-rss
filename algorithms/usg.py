import numpy as np
import pandas as pd
import recommenders as rec
import networkx as nx
import matplotlib.pyplot as plt
import areamanager
import math
import timeit
import geo_utils
from constants import geocat_constants
from constants import experiment_constants
from constants import usg_constants
from collections import defaultdict
from utils import transform_id_to_int
def string_to_array(string):
    if string == '':
        return list()
    return eval(string)
CITY=experiment_constants.get_city()

if __name__ == '__main__':
	df_city_checkin=pd.read_csv("../data/checkin/cities/train/"+CITY+".csv",converters={'categories':string_to_array})
	user_int_id,poi_int_id=transform_id_to_int(df_city_checkin)
	users_id=df_city_checkin['user_id'].drop_duplicates().reset_index(drop=True)
	user_num=int(len(user_int_id)/2)
	poi_num=int(len(poi_int_id)/2)
	df_city_checkin=df_city_checkin.set_index('user_id')

	training_matrix = np.zeros((user_num, poi_num))
	for user_id,poi_id in df_city_checkin.business_id.iteritems():
		training_matrix[user_id,poi_id]=1
	sim=training_matrix.dot(training_matrix.T)
	norms=[np.linalg.norm(training_matrix[i]) for i in range(training_matrix.shape[0])]
	for i in range(training_matrix.shape[0]):
		sim[i][i] = 0.0
		for j in range(i+1, training_matrix.shape[0]):
			sim[i][j] /= (norms[i] * norms[j])
			sim[j][i] /= (norms[i] * norms[j])
	social_relations = defaultdict(list)
	df_city_user=pd.read_csv("../data/user/"+CITY+".csv",converters={'friends':string_to_array})


