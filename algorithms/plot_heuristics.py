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
  # inquirer.Checkbox('heuristics',
  #                   message="Heuristics to execute",
  #                   choices=HEURISTICS,
  #                   ),

]

answers = inquirer.prompt(questions)
city = answers['city']

rr = RecRunner.getInstance("usg", "geocat", city, 80, 20,
        "../data",{})
rr.show_heuristic = True
# rr.print_latex_metrics_table(cities=answers['city'],prefix_name='heuristics',heuristic=True)
# rr.plot_bar_exclusive_metrics(prefix_name='heuristics',ncol=1)
rr.load_metrics(base=True,name_type=NameType.PRETTY)
for heuristic in ['local_max','tabu_search','particle_swarm']:
    rr.final_rec_parameters = {'heuristic': heuristic}
    rr.load_metrics(base=False,name_type=NameType.PRETTY)
rr.plot_maut(prefix_name='maut_heuristics',ncol=3)
