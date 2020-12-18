
import sys, os
sys.path.insert(0, os.path.abspath('lib'))
from lib.RecRunner import RecRunner
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
# rr.load_base()

# rr.run_base_recommender()
# rr.run_base_recommender()
# rr.run_all_final()
# rr=RecRunner("usg","geodiv","madison",80,20,"/home/heitor/recsys/data",final_rec_parameters={'cat_div_method':'ld'})


# rr.load_base()
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

# rr.final_rec= 'ld'
# rr.load_final_predicted()
# rr.eval_rec_metrics()

# rr=RecRunner("usg","geocat","lasvegas",80,20,"/home/heitor/recsys/data")
# rr.load_base()

# rr.load_base()

# rr.load_final_predicted()

# rr.eval_rec_metrics()
# rr.load_base()
# rr.run_base_recommender()

#rr.load_metrics(base=True)
# rr.final_rec_parameters = {'cat_div_method': 'ld'}
# rr.load_metrics(base=False)
# rr.set_final_rec_parameters({'cat_div_method':'std_norm'})
# rr.load_metrics(base=False)
rr.load_metrics(base=True)
for rec in ['gc','ld','geodiv','geocat']:
    rr.final_rec = rec
    rr.load_metrics(base=False)
# rr.final_rec_parameters = {'obj_func': 'og'}
# rr.load_metrics(base=False)
# rr.final_rec_parameters = {'obj_func': 'no_ild'}
# rr.load_metrics(base=False)
# rr.final_rec_parameters = {'obj_func': 'gc_diff'}
# rr.load_metrics(base=False)
# rr.final_rec_parameters = {'obj_func': 'div_2'}
# rr.load_metrics(base=False)
# rr.final_rec="perfectpersongeocat"
# rr.load_metrics(base=False)
rr.print_latex_metrics_table()
