
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
  inquirer.Checkbox('baser',
                    message="Base recommender",
                    choices=list(RecRunner.get_base_parameters().keys()),
                    ),
]

answers = inquirer.prompt(questions)
city = answers['city']
basers = answers['baser']

rr=RecRunner.getInstance("xxxx","geocat",city,80,20,"/home/heitor/recsys/data")
rr.load_base()
for baser in basers:
    rr.base_rec = baser
    rr.load_base_predicted()
    rr.eval_rec_metrics(base=True)

