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
]

answers = inquirer.prompt(questions)
city = answers['city']

rr = RecRunner.getInstance("usg", "perfectpersongeocat", city, 80, 20,
        "/home/heitor/recsys/data",{})
rr.perfect_analysis()
