import sys
import os
sys.path.insert(0, os.path.abspath('..'))
from library.RecRunner import RecRunner
from library.constants import experiment_constants,DATA

import app_utils
import argparse

argparser =argparse.ArgumentParser()
app_utils.add_cities_arg(argparser)
app_utils.add_base_recs_arg(argparser)
app_utils.add_final_recs_arg(argparser)
args = argparser.parse_args()


rr = RecRunner(args.base_recs[0], args.final_recs[0], args.cities[0], experiment_constants.N, experiment_constants.K,
                           DATA)

for city in args.cities:
  rr.city= city
  rr.load_base()
  for base_rec in args.base_recs:
    rr.base_rec = base_rec
    rr.load_base_predicted()
    for finalr in args.final_recs:
        rr.final_rec = finalr
        rr.run_final_recommender()
