import sys, os
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
rr=RecRunner.getInstance("usg","perfectpgeocat",city,80,20,
                         "../data")
rr.load_base(user_data=True)
rr.print_correlation_perfectparam()
