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
  inquirer.List('city',
                    message="City to use",
                    choices=experiment_constants.CITIES,
                    ),

]

answers = inquirer.prompt(questions)
city = answers['city']

rr = RecRunner.getInstance("usg", "geocat", city, 80, 20,
        "../data",{})
rr.load_base()
rr.cluster_methods()
