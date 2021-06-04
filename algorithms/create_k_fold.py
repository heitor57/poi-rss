import sys
import os
sys.path.insert(0, os.path.abspath('lib'))
from lib.RecRunner import RecRunner
# rr=RecRunner("usg","geocat","madison",80,20,"../data")
# print(rr.get_base_rec_file_name())
# print(rr.get_final_rec_file_name())

# rr.load_base()
# rr.run_base_recommender()
# rr.run_final_recommender()
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


rr = RecRunner.getInstance('usg', "persongeocat", city, 80, 20,
               "../data")

rr.load_base()
rr.k_fold = 3
rr.create_k_fold()


# rr.city = "charlotte"
# rr.load_base()
# rr.run_base_recommender()

# rr.load_final_predicted()
# rr.eval_rec_metrics()

