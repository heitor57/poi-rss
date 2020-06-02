import sys
import os
sys.path.insert(0, os.path.abspath('lib'))
from lib.RecRunner import RecRunner
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
  inquirer.Checkbox('heuristics',
                    message="Heuristics to execute",
                    choices=HEURISTICS,
                    ),

]

answers = inquirer.prompt(questions)
city = answers['city']

rr = RecRunner.getInstance("usg", "geocat", city, 80, 5,
        "/home/heitor/recsys/data",{})


rr.load_base()
rr.load_base_predicted()
i=0
for heuristic in answers['heuristics']:
    print("%s [%d/%d]"%(heuristic,i,len(answers['heuristics'])))
    rr.final_rec_parameters = {'heuristic': heuristic}
    rr.run_final_recommender()
    i+=1

