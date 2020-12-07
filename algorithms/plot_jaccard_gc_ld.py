import sys, os
sys.path.insert(0, os.path.abspath('lib'))
from lib.RecRunner import RecRunner
import inquirer
from lib.constants import experiment_constants

questions = [
  inquirer.List('city',
                    message="City to use",
                    choices=experiment_constants.CITIES,
                    ),
]

answers = inquirer.prompt(questions)
city = answers['city']

rr=RecRunner("usg","xxxx",city,80,20,"/home/heitor/recsys/data")

rr.plot_jaccard_ild_gc_correlation('usg','gc','ld',range(1,21))