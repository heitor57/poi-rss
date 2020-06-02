import sys
import os
sys.path.insert(0, os.path.abspath('lib'))
from lib.RecRunner import RecRunner
from lib.RecRunner import RECLIST
from lib.constants import experiment_constants
import inquirer
from colorama import Fore
from colorama import Style
from lib.geocat.objfunc import OBJECTIVE_FUNCTIONS
import os.path
questions = [
  inquirer.List('city',
                    message="City to use",
                    choices=experiment_constants.CITIES,
                    ),
  inquirer.List('baser',
                    message="Base recommender",
                    choices=list(RecRunner.get_base_parameters().keys()),
                    ),
  inquirer.Checkbox('item',
                    message="Objective function to execute",
                    choices=list(OBJECTIVE_FUNCTIONS.keys()),
                    ),

]

answers = inquirer.prompt(questions)
city = answers['city']
baser = answers['baser']
rr = RecRunner.getInstance(baser, "geocat", city, 80, 20,
        "/home/heitor/recsys/data",{})


rr.load_base()
i=1
for item in answers['item']:
    print("%s [%d/%d]"%(item,i,len(answers['item'])))
    rr.final_rec_parameters = {'obj_func': item}
    rr.load_final_predicted()
    rr.eval_rec_metrics()
    i+=1

