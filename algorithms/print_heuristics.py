import sys
import os
sys.path.insert(0, os.path.abspath('lib'))
from lib.RecRunner import RecRunner, NameType
from lib.RecRunner import RECLIST
from lib.constants import experiment_constants
import inquirer
from colorama import Fore
from colorama import Style
from lib.geocat.objfunc import HEURISTICS
import os.path
questions = [
  inquirer.Checkbox('city',
                    message="City to use",
                    choices=experiment_constants.CITIES,
                    ),
]

answers = inquirer.prompt(questions)
city = answers['city']

rr = RecRunner.getInstance("usg", "geocat", city[0], 80, 20,
        "../data",{})

rr.show_heuristic = True
rr.print_latex_cities_metrics_table(city,references=['USG'],prefix_name='heuristics',heuristic=True)
