from collections import defaultdict
from constants import experiment_constants
import pickle
import npyscreen

def convert_city(CITY):
    # data_checkin_train = pickle.load(open("../data/checkin/train/"+CITY+".pickle","rb"))

# #Test load
    # ground_truth = defaultdict(set)
    # for checkin in pickle.load(open("../data/checkin/test/"+CITY+".pickle","rb")):
    #     ground_truth[checkin['user_id']].add(checkin['poi_id'])
    #Pois load
    poi_cats = {}
    for poi_id,poi in pickle.load(open("../../data/poi/"+CITY+".pickle","rb")).items():
        poi_cats[poi_id] = poi['categories']
    fpoi_cats = open("../../data/poi/"+CITY+".td", 'w')

    for poi_id in poi_cats:
        fpoi_cats.write(' '.join([str(poi_id),
                                ' '.join(poi_cats[poi_id])
        ])+'\n'
        )
    fpoi_cats.close()

class ProcessButton(npyscreen.ButtonPress):
    def whenPressed(self):
        convert_city(self.name)
        npyscreen.notify_confirm(self.name+" converted!",editw=1)

class FormObject(npyscreen.Form):
    def create(self):
        self.add(npyscreen.TitleText,value='Press Enter in the cities you want to convert',name=' ')
        for city in experiment_constants.CITIES:
            self.add(ProcessButton,name=city)
    def afterEditing(self):
        self.parentApp.setNextForm(None)
class App(npyscreen.NPSAppManaged):
    def onStart(self):
        self.addForm('MAIN',FormObject,name="Base converter to Trotta's base!")
    
if __name__ == '__main__':
    app=App().run()
