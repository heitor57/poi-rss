import sys, os
sys.path.insert(0, os.path.abspath('lib'))
from lib.RecRunner import RecRunner,NameType
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

rr=RecRunner("mostpopular","geocat",city,80,20,"/home/heitor/recsys/data")

rr.base_rec = "usg"
rr.load_metrics(base=False,name_type = NameType.SHORT)
for phi in [0.0,1.0]:
    rr.final_rec_parameters = {'div_geo_cat_weight':1.0,'div_cat_weight':phi}
    rr.load_metrics(base=False,name_type = NameType.SHORT)

rr.plot_maut(ncol=1)

