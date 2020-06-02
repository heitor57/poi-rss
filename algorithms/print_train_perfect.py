import sys
import os
sys.path.insert(0, os.path.abspath('lib'))
from lib.RecRunner import RecRunner, NameType
from lib.pgc.CatDivPropensity import CatDivPropensity
from lib.pgc.GeoDivPropensity import GeoDivPropensity
from colorama import Fore
from colorama import Style
import inquirer

rr = RecRunner.getInstance("usg", "perfectpgeocat", sys.argv[1], 80, 20,
        "/home/heitor/recsys/data",{})
rr.load_base()
rr.print_train_perfect()
