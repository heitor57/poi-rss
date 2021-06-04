import sys
import os
sys.path.insert(0, os.path.abspath('lib'))
from lib.RecRunner import RecRunner, NameType
from lib.pgc.CatDivPropensity import CatDivPropensity
from lib.pgc.GeoDivPropensity import GeoDivPropensity
from colorama import Fore
from colorama import Style
import inquirer

rr = RecRunner.getInstance("usg", "geocat", sys.argv[1], 80, 20,
        "../data",{})
rr.load_metrics(base=True)
rr.load_metrics(base=False)
rr.final_rec="persongeocat"


choices=[]

for cat_div_method in [None]+CatDivPropensity.METHODS:
    for geo_div_method in [None]+GeoDivPropensity.METHODS:
        if cat_div_method != geo_div_method:
            rr.final_rec_parameters = {'cat_div_method': cat_div_method, 'geo_div_method': geo_div_method}
            if os.path.exists(rr.get_file_name_metrics(False,5)):
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
    rr.load_metrics(base=False,name_type=NameType.SHORT)
    i+=1

rr.print_latex_metrics_table(prefix_name='persons')
