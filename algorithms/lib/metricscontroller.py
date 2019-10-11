import numpy as np

class MetricsController:
    def __init__(self,metrics,algorithm,metrics_path):
        self.metrics={}
        for i in metrics:
            self.metrics[i]=[]
        self.algorithm=algorithm
        self.metrics_path=metrics_path
    def append_data(self,data):
        c=0
        for i in self.metrics.keys():
            self.metrics[i].append(data[c])
            c+=1
    def print_metrics(self):
        print(self.algorithm)
        for i in self.metrics.keys():
            print("mean(%s)=%.2f" % (i,np.mean(self.metrics[i])))