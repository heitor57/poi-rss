from collections import defaultdict
from constants import experiment_constants
import pickle
import npyscreen

def convert_city(CITY):
    data_checkin_train = pickle.load(open("../../data/checkin/train/"+CITY+".pickle","rb"))
    fcheckin_train = open("../../data/checkin/train/"+CITY+".td", 'w')
    for checkin in data_checkin_train:
        fcheckin_train.write('::'.join([str(checkin['user_id']),str(checkin['poi_id']),"0","0.0"])+'\n')
    fcheckin_train.close()
# #Test load
    data_checkin_test = pickle.load(open("../../data/checkin/test/"+CITY+".pickle","rb"))
        
    fcheckin_test = open("../../data/checkin/test/"+CITY+".td", 'w')

    for checkin in data_checkin_test:
        fcheckin_test.write('::'.join([str(checkin['user_id']),str(checkin['poi_id']),"0","0.0"])+'\n')

    fcheckin_test.close()
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
        for city in experiment_constants.CITIES:
            self.add(ProcessButton,name=city)
    def afterEditing(self):
        self.parentApp.setNextForm(None)
class App(npyscreen.NPSAppManaged):
    def onStart(self):
        self.addForm('MAIN',FormObject,name="Base converter to Trotta's base!")
    
if __name__ == '__main__':
    app=App().run()
