import sys
import os

from numpy.core.fromnumeric import argpartition
sys.path.insert(0, os.path.abspath('..'))
from library.RecRunner import RecRunner

from library.constants import DATA
import app_utils
import argparse
argparser =argparse.ArgumentParser()
argparser.add_argument(app_utils.ARG_CITIES)
argparser.add_argument(app_utils.ARG_BASE_RECS)
args = argparser.parse_args()
# questions = [
  # inquirer.Checkbox('cities',
                    # message="City to use",
                    # choices=experiment_constants.CITIES,
                    # ),
  # inquirer.Checkbox('baser',
                    # message="Base recommender",
                    # choices=list(RecRunner.get_base_parameters().keys()),
                    # ),
# ]

# answers = inquirer.prompt(questions)
# # baser = answers['baser']
# base_recommenders = answers['baser']


rr = RecRunner.getInstance(base_recommenders[0], "persongeocat", "madison", 80, 20,
              DATA)

for city in answers['cities']:
  rr.city= city
  rr.load_base()
  for baser in base_recommenders:
      rr.base_rec = baser
      rr.run_base_recommender()


# rr.city = "charlotte"
# rr.load_base()
# rr.run_base_recommender()

# rr.load_final_predicted()
# rr.eval_rec_metrics()

