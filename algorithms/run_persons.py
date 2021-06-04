import sys
import os
sys.path.insert(0, os.path.abspath('lib'))
from lib.RecRunner import RecRunner
from lib.RecRunner import RECLIST
from lib.pgc.CatDivPropensity import CatDivPropensity
from lib.pgc.GeoDivPropensity import GeoDivPropensity
from lib.constants import experiment_constants
import inquirer
from colorama import Fore
from colorama import Style
import os.path
questions = [
  inquirer.List('city',
                    message="City to use",
                    choices=experiment_constants.CITIES,
                    ),
]

answers = inquirer.prompt(questions)
city = answers['city']

rr = RecRunner.getInstance("usg", "persongeocat", city, 80, 20,
        "../data",{})


rr.load_base()
rr.load_base_predicted()
choices=[]
for cat_div_method in [None]+CatDivPropensity.METHODS:
    for geo_div_method in [None]+GeoDivPropensity.METHODS:
        if cat_div_method != None or geo_div_method != None:
            rr.final_rec_parameters = {'cat_div_method': cat_div_method, 'geo_div_method': geo_div_method}
            if os.path.exists(rr.data_directory+RECLIST+rr.get_final_rec_file_name()):
                choices.append((f"{Fore.GREEN}{cat_div_method} and {geo_div_method}{Style.RESET_ALL}" ,[cat_div_method,geo_div_method]))
            else:
                choices.append((f"{cat_div_method} and {geo_div_method}" ,[cat_div_method,geo_div_method]))


questions = [
  inquirer.Checkbox('methods',
                    message="Methods to execute",
                    choices=choices,
                    ),
]
answers = inquirer.prompt(questions)
i=0
for method in answers['methods']:
    print("%s [%d/%d]"%(method,i,len(answers['methods'])))
    cat_div_method = method[0]
    geo_div_method = method[1]
    rr.final_rec_parameters = {'cat_div_method': cat_div_method, 'geo_div_method': geo_div_method}
    rr.run_final_recommender()
    i+=1


