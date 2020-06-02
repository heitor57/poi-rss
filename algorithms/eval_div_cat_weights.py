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

rr = RecRunner.getInstance("usg", "geocat", city, 80, 20,
                           "/home/heitor/recsys/data")
rr.load_base()

x = 1
r = 0.1
for i in np.around(np.append(np.arange(0, x, r),x),decimals=2):
    rr.final_rec_parameters={'div_cat_weight':i}
    rr.load_final_predicted()
    rr.eval_rec_metrics()
