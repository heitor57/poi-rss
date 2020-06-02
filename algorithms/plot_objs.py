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
  inquirer.Checkbox('item',
                    message="Objective function to execute",
                    choices=list(OBJECTIVE_FUNCTIONS.keys()),
                    ),

]

answers = inquirer.prompt(questions)
city = answers['city']

rr = RecRunner.getInstance("usg", "geocat", city, 80, 20,
        "/home/heitor/recsys/data",{})

rr.load_metrics(base=True)

for rec in ['ld','binomial','pm2','geodiv']:
    rr.final_rec = rec
    rr.load_metrics(base=False)
rr.final_rec = 'geocat'
i=1
for item in answers['item']:
    print("%s [%d/%d]"%(item,i,len(answers['item'])))
    rr.final_rec_parameters = {'obj_func': item}
    rr.load_metrics(base=False)
    i+=1

rr.plot_bar_exclusive_metrics(prefix_name='objs',ncol=1)
rr.plot_maut(prefix_name='maut_objs',ncol=1)
rr.print_latex_metrics_table('objs')
