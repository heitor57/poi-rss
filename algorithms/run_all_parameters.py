import sys
import os
sys.path.insert(0, os.path.abspath('lib'))
from lib.RecRunner import RecRunner
import numpy as np
from lib.constants import experiment_constants
import inquirer

questions = [
  inquirer.List('city',
                    message="City to use",
                    choices=experiment_constants.CITIES,
                    ),
  inquirer.List('baser',
                    message="Base recommender",
                    choices=list(RecRunner.get_base_parameters().keys()),
                    ),
]

answers = inquirer.prompt(questions)
city = answers['city']
baser = answers['baser']


# rr=RecRunner("usg","geocat","madison",80,20,"/home/heitor/recsys/data")
# print(rr.get_base_rec_file_name())
# print(rr.get_final_rec_file_name())

# rr.load_base()
# rr.run_base_recommender()
# rr.run_final_recommender()

rr = RecRunner.getInstance(baser, "geocat", city, 80, 10,
                           "/home/heitor/recsys/data")
rr.load_base()
rr.load_base_predicted()

l_cat = np.sort(np.append(np.around(np.linspace(0, 1, 6),decimals=2),[0.05,0.1,0.9,0.95]))
l_geocat_div = np.around(np.linspace(0,1,5),decimals=2)
for div_weight in l_geocat_div:
    rr.final_rec_parameters['div_weight'] = div_weight
    for div_geo_cat_weight in l_geocat_div:
        rr.final_rec_parameters['div_geo_cat_weight'] = div_geo_cat_weight
        for div_cat_weight in l_cat:
            rr.final_rec_parameters['div_cat_weight'] = div_cat_weight
            if not(div_weight==0.0 and (div_geo_cat_weight!=div_weight or div_cat_weight!=div_weight)):
                rr.run_final_recommender(check_already_exists=False)
