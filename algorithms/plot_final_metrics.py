
import sys, os
sys.path.insert(0, os.path.abspath('lib'))
from lib.RecRunner import RecRunner
from lib.RecRunner import NameType
# rr=RecRunner("usg","geocat","madison",80,20,"/home/heitor/recsys/data")
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
  inquirer.List('baser',
                    message="Base recommender",
                    choices=list(RecRunner.get_base_parameters().keys()),
                    ),
]

answers = inquirer.prompt(questions)
city = answers['city']
baser = answers['baser']
rr=RecRunner(baser,"geocat",city,80,20,"/home/heitor/recsys/data")

rr.load_metrics(base=True,name_type=NameType.PRETTY)
for rec in ['gc','ld','binomial','pm2','geodiv','geocat','geodiv2020']:
    rr.final_rec = rec
    rr.load_metrics(base=False,name_type=NameType.PRETTY)
# rr.plot_bar_exclusive_metrics()
rr.plot_gain_metrics()
rr.plot_maut()
# rr.print_latex_metrics_table(prefix_name='all_met')
