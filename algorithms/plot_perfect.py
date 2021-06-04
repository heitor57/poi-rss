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
  inquirer.List('finalr',
                    message="Final recommender",
                    choices=['perfectpgeocat','pdpgeocat'],
                    ),
]

answers = inquirer.prompt(questions)
city = answers['city']
finalr = answers['finalr']

rr = RecRunner.getInstance("usg", finalr, city, 80, 20,
        "../data",{})


rr.plot_perfect_parameters()
