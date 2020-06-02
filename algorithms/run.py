import sys
import os
sys.path.insert(0, os.path.abspath('lib'))
from lib.RecRunner import RecRunner
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
  inquirer.Checkbox('finalr',
                    message="Final recommender",
                    choices=list(RecRunner.get_final_parameters().keys()),
                    ),
]

answers = inquirer.prompt(questions)
city = answers['city']
# finalr = answers['finalr']
baser = answers['baser']


rr = RecRunner.getInstance(baser, "geocat", city, 80, 20,
                           "/home/heitor/recsys/data")
rr.load_base()
rr.load_base_predicted()
for finalr in answers['finalr']:
    rr.final_rec = finalr
    rr.run_final_recommender()
