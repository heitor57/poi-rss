
import sys, os
sys.path.insert(0, os.path.abspath('lib'))
from lib.RecRunner import RecRunner
from lib.constants import experiment_constants
import inquirer
# rr=RecRunner("usg","geocat","madison",80,20,"../data")
# print(rr.get_base_rec_file_name())
# print(rr.get_final_rec_file_name())

# rr.load_base()
# rr.run_base_recommender()
# rr.run_final_recommender()
questions = [
  inquirer.List('city',
                    message="City to use",
                    choices=experiment_constants.CITIES,
                    ),
  inquirer.List('baser',
                    message="Base recommender",
                    choices=list(RecRunner.get_base_parameters().keys()),
                    ),
  inquirer.Checkbox('finalr',
                    message="Final recommender",
                    choices=list(RecRunner.get_final_parameters().keys()),
                    ),
]

answers = inquirer.prompt(questions)
city = answers['city']
baser = answers['baser']
finalr = answers['finalr']

rr=RecRunner.getInstance(baser,'xxx',city,80,20,"../data")
rr.load_base()
for fr in finalr:
    rr.final_rec = fr
    rr.load_final_predicted()
    rr.eval_rec_metrics(METRICS_KS=range(2,21))
