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
]

answers = inquirer.prompt(questions)
city = answers['city']


# rr=RecRunner("usg","geocat","madison",80,20,"../data")
# print(rr.get_base_rec_file_name())
# print(rr.get_final_rec_file_name())

# rr.load_base()
# rr.run_base_recommender()
# rr.run_final_recommender()

rr = RecRunner.getInstance("usg", "geocat", city, 80, 20,
                           "../data")
rr.load_base()
rr.load_base_predicted()

x = 1
r = 0.25
for i in np.append(np.arange(0, x, r),x):
    for j in np.append(np.arange(0, x, r),x):
        if not(i==0 and i!=j):
            rr.final_rec_parameters={'div_weight':i,'div_geo_cat_weight':j}
            rr.run_final_recommender()
