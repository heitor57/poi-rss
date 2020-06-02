
import sys, os
sys.path.insert(0, os.path.abspath('lib'))
from lib.RecRunner import RecRunner
# rr=RecRunner("usg","geocat","madison",80,20,"/home/heitor/recsys/data")
# print(rr.get_base_rec_file_name())
# print(rr.get_final_rec_file_name())

# rr.load_base()
# rr.run_base_recommender()
# rr.run_final_recommender()

import inquirer
from colorama import Fore
from colorama import Style
from lib.constants import experiment_constants

questions = [
  inquirer.List('city',
                    message="City to use",
                    choices=experiment_constants.CITIES,
                    ),
  inquirer.List('baser',
                    message="Base recommender",
                    choices=list(RecRunner.get_base_parameters().keys()),
                    ),
]

answers = inquirer.prompt(questions)
city = answers['city']
baser = answers['baser']

rr=RecRunner.getInstance(baser,"perfectpersongeocat",city,80,20,"/home/heitor/recsys/data")
rr.load_base()
rr.load_base_predicted()
rr.eval_rec_metrics(base=True)
rr.run_all_eval()
