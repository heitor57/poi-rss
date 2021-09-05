import numpy as np

class MetricsController:
    def __init__(self,metrics=None,algorithm="generic",metrics_path="/home/heitor/recsys/data/metrics",k=10):
        self.algorithm=algorithm
        self.metrics_path=metrics_path
        self.k=k
        
        if metrics==None:
            self.fload_metrics()
        else:
            self.metrics={}
            for i in metrics:
                self.metrics[i]={}
    def append_data(self,uid,data):
        c=0
        for i in self.metrics.keys():
            self.metrics[i][uid]=data[c]
            c+=1
            
    def get_metrics_mean(self):
        result=str()
        for c,i in enumerate(self.metrics.keys()):
            result+=("%s→%.6f"+ ('' if (c==len(self.metrics.keys())-1) else ',')) % (i,np.mean(self.metrics[i]))
        return result
    def print_metrics(self):
        #for c,i in enumerate(self.metrics.keys()):
          #  print("%s→%.6f" % (i,np.mean(self.metrics[i])),end=((c==len(self.metrics.keys())-1)?'':','))
        print(self.get_metrics_mean())
        pass
    
    def __str__(self):
        return f"Metrics-{list(self.metrics.keys())}\nAlgorithm-{self.algorithm}\nMetricsPath-{self.metrics_path}\nRecSize-{self.k}\n"+self.get_metrics_mean()
    
    def fwrite_metrics(self):
        fname=self.metrics_path+"/"+self.algorithm+"_at_"+str(self.k)
        result=str()
        result=f"""{str(self.metrics)}"""
#         for c,i in enumerate(self.metrics.keys()):
#             result+=("%s\t%.6f" % (i,np.mean(self.metrics[i]))+('' if (c==len(self.metrics.keys())-1) else '\n'))       
        f=open(fname,"w+")
        f.write(result)
        f.close()
        print("File "+fname+" written with success")
    def fload_metrics(self):
        fname=self.metrics_path+"/"+self.algorithm+"_at_"+str(self.k)
        f=open(fname,"r")
        self.metrics=eval(f.read())

        f.close()