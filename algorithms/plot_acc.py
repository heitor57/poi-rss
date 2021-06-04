import sys, os
sys.path.insert(0, os.path.abspath('lib'))
from lib.RecRunner import RecRunner
import inquirer
from colorama import Fore
from colorama import Style
from lib.constants import experiment_constants

questions = [
  inquirer.List('city',
                    message="City to use",
                    choices=experiment_constants.CITIES,
                    ),
]

answers = inquirer.prompt(questions)
city = answers['city']
rr=RecRunner("usg","persongeocat",city,80,20,"../data")

rr.load_metrics(base=True)
rr.load_metrics(base=False)
rr.final_rec="geodiv"
rr.load_metrics(base=False)
rr.final_rec="geocat"
rr.load_metrics(base=False)
rr.final_rec="ld"
rr.load_metrics(base=False)
rr.final_rec="binomial"
rr.load_metrics(base=False)
rr.final_rec="pm2"
rr.load_metrics(base=False)
rr.final_rec="perfectpersongeocat"
rr.load_metrics(base=False)
rr.plot_acc_metrics()
