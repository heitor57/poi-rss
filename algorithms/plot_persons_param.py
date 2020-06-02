import sys
import os
sys.path.insert(0, os.path.abspath('lib'))
from lib.RecRunner import RecRunner
from lib.pgc.CatDivPropensity import CatDivPropensity
from lib.pgc.GeoDivPropensity import GeoDivPropensity
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

rr = RecRunner.getInstance("usg", "persongeocat", city, 80, 20,
        "/home/heitor/recsys/data",{})

choices=[]
for cat_div_method in [None]+CatDivPropensity.METHODS:
    for geo_div_method in [None]+GeoDivPropensity.METHODS:
        if cat_div_method != None or geo_div_method != None:
            choices.append([cat_div_method,geo_div_method])


questions = [
  inquirer.Checkbox('methods',
                    message="Methods to execute",
                    choices=choices,
                    ),
]
answers = inquirer.prompt(questions)

rr.load_base()
i=0
for method in answers['methods']:
    print("%s [%d/%d]"%(method,i,len(answers['methods'])))
    cat_div_method = method[0]
    geo_div_method = method[1]
    rr.final_rec_parameters = {'cat_div_method': cat_div_method, 'geo_div_method': geo_div_method}
    rr.plot_personparameter()
    i+=1
