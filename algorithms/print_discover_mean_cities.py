import sys, os
sys.path.insert(0, os.path.abspath('lib'))
from lib.RecRunner import RecRunner
import inquirer
from lib.constants import experiment_constants


rr=RecRunner('usg',"geocat",'xxx',80,20,"../data")

rr.print_discover_mean_cities()
