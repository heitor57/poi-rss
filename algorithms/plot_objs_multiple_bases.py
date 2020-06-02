import sys
import os
sys.path.insert(0, os.path.abspath('lib'))
from lib.RecRunner import RecRunner
from lib.RecRunner import RECLIST
from lib.constants import experiment_constants, RECS_PRETTY
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
  inquirer.Checkbox('item',
                    message="Objective function to execute",
                    choices=list(OBJECTIVE_FUNCTIONS.keys()),
                    ),

]

answers = inquirer.prompt(questions)
city = answers['city']

rr = RecRunner.getInstance("usg", "geocat", city, 80, 20,
        "/home/heitor/recsys/data",{})

i=1
base_recs = ['usg','geosoca']
for base_rec in base_recs:
    rr.base_rec = base_rec
    rr.load_metrics(base=True,pretty_name=True)
    for item in answers['item']:
        print("%s [%d/%d]"%(item,i,len(base_recs)*len(answers['item'])))
        rr.final_rec_parameters = {'obj_func': item}
        rr.load_metrics(base=False,pretty_name=True,pretty_with_base_name=True)
        i+=1

rr.plot_bar_exclusive_metrics(prefix_name='objs_bases',ncol=1)
rr.print_latex_metrics_table('objs_bases',references = list(map(RECS_PRETTY.get,['usg','geosoca'])))
