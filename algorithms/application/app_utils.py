import sys
import os
sys.path.insert(0, os.path.abspath('..'))
from library.RecRunner import RecRunner
from library.constants import experiment_constants
import argparse
def add_cities_arg(parser:argparse.ArgumentParser):
    parser.add_argument('--cities',nargs='*',help=f'City, e.g., {", ".join(experiment_constants.CITIES)}')
def add_base_recs_arg(parser:argparse.ArgumentParser):
    parser.add_argument('--base_recs',nargs='*',help=f'Base recommender, e.g., {", ".join(list(RecRunner.get_base_parameters().keys()))}')
def add_final_recs_arg(parser:argparse.ArgumentParser):
    parser.add_argument('--final_recs',nargs='*',help=f'Final recommenders, e.g., {", ".join(list(RecRunner.get_final_parameters().keys()))}')

# def add_city_arg(parser:argparse.ArgumentParser):
    # parser.add_argument('--city',help=f'City, e.g., {", ".join(experiment_constants.CITIES)}')
# def add_base_rec_arg(parser:argparse.ArgumentParser):
    # parser.add_argument('--base_rec',help=f'Base recommender, e.g., {", ".join(list(RecRunner.get_base_parameters().keys()))}')
# def add_final_rec_arg(parser:argparse.ArgumentParser):
    # parser.add_argument('--final_rec',help=f'Final recommender, e.g., {", ".join(list(RecRunner.get_final_parameters().keys()))}')
