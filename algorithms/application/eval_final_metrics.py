import os
import sys
sys.path.insert(0, os.path.abspath('..'))
import inquirer
from library.constants import experiment_constants,DATA
from library.RecRunner import RecRunner
import sys

questions = [
    inquirer.Checkbox('city',
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
cities = answers['city']
baserecs = answers['baser']
finalrecs = answers['finalr']

rr = RecRunner.getInstance(
    baserecs[0], finalrecs[0], cities[0], 80, 20, DATA)
for city in cities:
    rr.city = city
    rr.load_base()
    for baserec in baserecs:
        rr.base_rec = baserec
        for finalrec in finalrecs:
            rr.final_rec = finalrec
            rr.load_final_predicted()
            rr.eval_rec_metrics()
