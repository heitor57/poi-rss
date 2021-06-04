import sys
import os
sys.path.insert(0, os.path.abspath('lib'))
from lib.RecRunner import RecRunner
from lib.constants import experiment_constants
import inquirer

rr = RecRunner.getInstance("usg", "persongeocat", 'lasvegas', 80, 20,
        "../data",{})

rr.plot_cities_cdf()
