import sys, os
sys.path.insert(0, os.path.abspath('lib'))
from lib.RecRunner import RecRunner, NameType
import inquirer
from lib.constants import experiment_constants

city= 'lasvegas'
rr=RecRunner("usg","geocat",city,80,20,"../data")

rr.load_metrics(base=False)
rr.final_rec="perfectpgeocat"
rr.load_metrics(base=False)
train_sizes = [0.7,0.8,0.85,0.9]
for train_size in train_sizes:
    self.final_rec_parameters['train_size'] = train_size
    rr.load_metrics(base=False,name_type=NameType.FULL)

rr.plot_gain_metric(prefix_name='optimal')

