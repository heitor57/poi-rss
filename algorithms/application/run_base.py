import sys
import os

from numpy.core.fromnumeric import argpartition
sys.path.insert(0, os.path.abspath('..'))
from library.RecRunner import RecRunner

from library.constants import DATA, experiment_constants
import app_utils
import argparse

argparser =argparse.ArgumentParser()
app_utils.add_cities_arg(argparser)
app_utils.add_base_recs_arg(argparser)
args = argparser.parse_args()

rr = RecRunner(args.base_recs[0], "none", args.cities[0], experiment_constants.N, experiment_constants.K,
              DATA)

for city in args.cities:
  rr.city= city
  rr.load_base()
  for baser in args.base_recs:
      rr.base_rec = baser
      rr.run_base_recommender()
