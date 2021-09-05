import sys
import os
sys.path.insert(0, os.path.abspath('..'))
# import argparse
from library.RecRunner import RecRunner
from library.constants import experiment_constants,DATA
# CHECKBOX_CITIES = inquirer.Checkbox('cities',
                # message="City to use",
                # choices=experiment_constants.CITIES,
                # )

# CHECKBOX_BASE_RECS =inquirer.Checkbox('base_recs',
                # message="Base recommenders",
                # choices=list(RecRunner.get_base_parameters().keys()),
                # )

# CHECKBOX_FINAL_RECS =inquirer.Checkbox('final_recs',
                # message="Final recommenders",
                # choices=list(RecRunner.get_final_parameters().keys()))

ARG_CITIES =  {'name_or_flags':'-c','nargs':'*','help':f'Cities, e.g., {", ".join(experiment_constants.CITIES)}'}
ARG_BASE_RECS =  {'name_or_flags':'-b','nargs':'*','help':f'Base recommenders, e.g., {", ".join(list(RecRunner.get_base_parameters().keys()))}'}
ARG_FINAL_RECS =  {'name_or_flags':'-f','nargs':'*','help':f'Final recommenders, e.g., {", ".join(list(RecRunner.get_final_parameters().keys()))}'}

# a = argparse.ArgumentParser()
# a.add_argument(name_or_flags=)
