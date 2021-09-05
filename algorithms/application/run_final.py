import sys
import os
sys.path.insert(0, os.path.abspath('..'))
from library.RecRunner import RecRunner
from library.constants import experiment_constants,DATA
import inquirer

questions = [
  inquirer.Checkbox('cities',
                    message="City to use",
                    choices=experiment_constants.CITIES,
                    ),
  inquirer.Checkbox('baser',
                    message="Base recommender",
                    choices=list(RecRunner.get_base_parameters().keys()),
                    ),
  inquirer.Checkbox('finalr',
                    message="Final recommender",
                    choices=list(RecRunner.get_final_parameters().keys()),
                    ),
]

answers = inquirer.prompt(questions)
cities = answers['cities']
# finalr = answers['finalr']
baser = answers['baser']


rr = RecRunner.getInstance(baser[0], "geocat", cities[0], 80, 20,
                           DATA)

for city in answers['cities']:
  rr.city= city
  rr.load_base()
  for base_rec in answers['baser']:
    rr.base_rec = base_rec
    rr.load_base_predicted()
    for finalr in answers['finalr']:
        rr.final_rec = finalr
        rr.run_final_recommender()
