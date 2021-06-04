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

rr = RecRunner.getInstance(baser, "binomial", city, 80, 10,
                           "../data")
rr.load_base()
rr.load_base_predicted()

lp = np.around(np.linspace(0, 1, 5),decimals=2)
for div_weight in lp:
    for alpha in lp:
        if not(div_weight == 0 and alpha != div_weight):
            rr.final_rec_parameters['div_weight'] = div_weight
            rr.final_rec_parameters['alpha'] = alpha
            rr.run_final_recommender(check_already_exists=False)
