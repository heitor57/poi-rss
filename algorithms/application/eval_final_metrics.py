import os
import sys
sys.path.insert(0, os.path.abspath('..'))
from library.constants import DATA,experiment_constants
from library.RecRunner import RecRunner
import sys
import app_utils
import argparse

argparser =argparse.ArgumentParser()
app_utils.add_cities_arg(argparser)
app_utils.add_base_recs_arg(argparser)
app_utils.add_final_recs_arg(argparser)
args = argparser.parse_args()

rr = RecRunner(
   args.base_recs[0], args.final_recs[0], args.cities[0], experiment_constants.N, experiment_constants.K, DATA)
for city in args.cities:
    rr.city = city
    rr.load_base()
    for baserec in args.base_recs:
        rr.base_rec = baserec
        for finalrec in args.final_recs:
            rr.final_rec = finalrec
            rr.load_final_predicted()
            rr.eval_rec_metrics()
