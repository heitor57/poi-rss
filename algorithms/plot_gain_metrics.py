
import sys, os
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

# questions = [
#   inquirer.List('city',
#                     message="City to use",
#                     choices=experiment_constants.CITIES,
#                     ),
# ]

# answers = inquirer.prompt(questions)
# city = answers['city']
city= 'x'
rr=RecRunner("mostpopular","persongeocat",city,80,20,"../data")

# rr.base_rec = "usg"
# rr.load_metrics(base=True)
# for rec in [# 'ld','binomial','pm2','geodiv',
#             'geocat']:
#     rr.final_rec = rec
#     rr.load_metrics(base=False)
rr.plot_adapt_gain_metrics(['lasvegas','phoenix'],['usg','geosoca'],
                           ['geocat'],ncol=1)

