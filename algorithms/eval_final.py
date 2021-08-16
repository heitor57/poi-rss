
import sys, os
sys.path.insert(0, os.path.abspath('lib'))
from lib.RecRunner import RecRunner
from lib.constants import experiment_constants
import inquirer
# rr=RecRunner("usg","geocat","madison",80,20,"../data")
# print(rr.get_base_rec_file_name())
# print(rr.get_final_rec_file_name())

# rr.load_base()
# rr.run_base_recommender()
# rr.run_final_recommender()
questions = [
  inquirer.Checkbox('city',
                    message="City to use",
                    choices=experiment_constants.CITIES,
                    ),
  inquirer.Checkbox('baser',
                    message="Base recommender",
                    choices=list(RecRunner.get_base_parameters().keys()),
                    ),
  inquirer.Checkbox('finalr',
                    message="Final recommender",
                    choices=list(RecRunner.get_final_parameters().keys()),
                    ),
]

answers = inquirer.prompt(questions)
cities = answers['city']
baserecs = answers['baser']
finalrecs = answers['finalr']

rr=RecRunner.getInstance(baserecs[0],finalrecs[0],cities[0],80,20,"../data")
for city in cities:
  rr.city= city
  rr.load_base()
  for baserec in baserecs:
    rr.base_rec = baserec
    for finalrec in finalrecs:
        rr.final_rec = finalrec
        rr.load_final_predicted()
        rr.eval_rec_metrics()
# rr.run_base_recommender()
# rr.run_base_recommender()
# rr.run_all_final()
# rr=RecRunner("usg","geodiv","madison",80,20,"../data",final_rec_parameters={'cat_div_method':'ld'})


# rr.load_base_predicted()
# rr.run_final_recommender()

# rr.load_final_predicted()
# rr.eval_rec_metrics()


# rr.final_rec= 'persongeocat'
# rr.load_final_predicted()
# rr.eval_rec_metrics()

# rr.final_rec= 'geodiv'
# rr.load_final_predicted()
# rr.eval_rec_metrics()

# rr=RecRunner("usg","geocat","lasvegas",80,20,"../data")
# rr.load_base()

# rr.load_base()

# rr.load_final_predicted()

# rr.eval_rec_metrics()
# rr.load_base()
# rr.run_base_recommender()

# rr.load_metrics(base=False)
# rr.load_metrics(base=True)
# rr.set_final_rec_parameters({'cat_div_method':'std_norm'})
# rr.load_metrics(base=False)
# rr.final_rec="geodiv"
# rr.load_metrics(base=False)
# rr.final_rec="geocat"
# rr.load_metrics(base=False)
# rr.base_rec="mostpopular"
# rr.load_metrics(base=True)

# rr.plot_bar_metrics()

