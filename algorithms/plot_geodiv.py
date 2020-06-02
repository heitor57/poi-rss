import sys
import os
sys.path.insert(0, os.path.abspath('lib'))
from lib.RecRunner import RecRunner
import numpy as np
# rr=RecRunner("usg","geocat","madison",80,20,"/home/heitor/recsys/data")
# print(rr.get_base_rec_file_name())
# print(rr.get_final_rec_file_name())

# rr.load_base()
# rr.run_base_recommender()
# rr.run_final_recommender()

rr = RecRunner.getInstance("usg", "persongeocat", "madison", 80, 20,
                           "/home/heitor/recsys/data")
rr.final_rec_parameters = {'geo_div_method': 'walk', 'cat_div_method':None}
rr.load_base()
rr.persongeocat_preprocess()
rr.plot_geopersonparameter()
rr.plot_geodivprop()
rr.plot_catdivprop()
rr.plot_users_max_min_catprop()
rr.plot_relation_catdivprop_catvisits()
rr.plot_geocatdivprop()
rr.plot_personparameter()
