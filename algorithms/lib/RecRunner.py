# import os
# import sys
# module_path = os.path.abspath(os.path.join('.'))
# if module_path not in sys.path:
#     sys.path.append(module_path)
# print(sys.path)

from abc import ABC, abstractmethod
from collections import defaultdict
import pickle
from concurrent.futures import ProcessPoolExecutor
import json
import time

import numpy as np
from progressbar import progressbar
from scipy.stats import describe
import matplotlib.pyplot as plt

import cat_utils
from usg.UserBasedCF import UserBasedCF
from usg.FriendBasedCF import FriendBasedCF
from usg.PowerLaw import PowerLaw
import geocat.objfunc as gcobjfunc
from pgc.GeoDivPropensity import GeoDivPropensity
from pgc.CatDivPropensity import CatDivPropensity
from constants import experiment_constants
import metrics


DATA_DIRECTORY='../data' # directory with all data

R_FORMAT='.json' # Results format is json, metrics, rec lists, etc
D_FORMAT='.pickle' # data set format is pickle

TRAIN='checkin/train' # train data sets
TEST='checkin/test' # test data sets
POI='poi/' # poi data sets with cats and coos
USER='user/' # users and friends
NEIGHBOR='neighbor/' # neighbors of pois

METRICS='result/metrics/'
METRICS='result/reclist/'

def normalize(scores):
    scores=np.array(scores,dtype=np.float64)
    max_score = np.max(scores)
    if not max_score == 0:
        scores = [s / max_score for s in scores]
    return scores
def dict_to_list_gen(d):
    for k,v in zip(d.keys(),d.values()):
        yield k
        yield v
def dict_to_list(d):
    return list(dict_to_list_gen(d))

class RecRunner:

    def __init__(self, base_rec, final_rec, city,
                 base_rec_list_size, final_rec_list_size, data_directory,
                 base_rec_parameters={},final_rec_parameters={}):
        self.BASE_RECOMMENDERS = {
            "mostpopular": self.mostpopular,
            "usg": self.usg
        }
        self.FINAL_RECOMMENDERS = {
            "geocat": self.geocat,
            "persongeocat": self.persongeocat,
            "geodiv" : self.geodiv
        }
        # self.BASE_RECOMMENDERS_PARAMETERS = {
        #     "mostpopular": [],
        #     "usg": ['alpha','beta','eta']
        # }
        # self.FINAL_RECOMMENDERS_PARAMETERS = {
        #     "geocat": ['div_weight','div_geo_cat_weight'],
        #     "persongeocat": ['div_weight']
        # }
        self.base_rec=base_rec
        self.final_rec=final_rec

        self.set_base_rec_parameters(base_rec_parameters)
        self.set_final_rec_parameters(final_rec_parameters)

        self.city = city

        self.base_rec_list_size = base_rec_list_size
        self.final_rec_list_size = final_rec_list_size

        self.data_directory = data_directory

        # buffers de resultado do metodo base
        self.user_base_predicted_lid={}
        self.user_base_predicted_score={}
        # buffers de resultado do metodo final
        self.user_final_predicted_lid={}
        self.user_final_predicted_score={}

        self.metrics={}
        self.metrics_name=['precision','recall','pr','ild','gc','epc']

    @property
    def data_directory(self):
        return self._data_directory
    @data_directory.setter
    def data_directory(self,data_directory):
        print("EWQ")
        if data_directory[-1] != '/':
            data_directory += '/'
        self._data_directory=data_directory
    @property
    def base_rec(self):
        return self._base_rec
    @base_rec.setter
    def base_rec(self,base_rec):
        if base_rec not in self.BASE_RECOMMENDERS:
            self._base_rec = next(iter(self.BASE_RECOMMENDERS))
            print(f"Base recommender not detected, using default:{self._base_rec}")
        else:
            self._base_rec = base_rec
        # parametros para o metodo base
        self.set_base_rec_parameters({})

    @property
    def final_rec(self):
        return self._final_rec
    @final_rec.setter
    def final_rec(self,final_rec):
        if final_rec not in self.FINAL_RECOMMENDERS:
            self._final_rec = next(iter(self.FINAL_RECOMMENDERS))
            print(f"Base recommender not detected, using default:{self._final_rec}")
        else:
            self._final_rec = final_rec
        # parametros para o metodo final
        self.set_final_rec_parameters({})
    
    def set_final_rec_parameters(self,parameters):
        final_parameters=self.get_final_parameters()[self.final_rec]
        for parameter in parameters.copy():
            if parameter not in final_parameters:
                del parameters[parameter]
        
        for parameter in final_parameters:
            if parameter not in parameters:
                parameters[parameter]=final_parameters[parameter]
        self.final_rec_parameters=parameters
    def set_base_rec_parameters(self,parameters):
        base_parameters=self.get_base_parameters()[self.base_rec]
        for parameter in parameters.copy():
            if parameter not in base_parameters:
                del parameters[parameter]
        
        for parameter in base_parameters:
            if parameter not in parameters:
                parameters[parameter]=base_parameters[parameter]
        self.base_rec_parameters=parameters
    
    @staticmethod
    def get_base_parameters():
        return {
            "mostpopular": {},
            "usg": {'alpha':0.1,'beta':0.1,'eta':0.05}
        }
    @staticmethod
    def get_final_parameters():
        return  {
            "geocat": {'div_weight':0.75,'div_geo_cat_weight':0.75},
            "persongeocat": {'div_weight':0.75,'cat_div_method':'ld'},
            "geodiv": {'div_weight':0.75}
        }

    def get_base_rec_name(self):
        list_parameters=list(map(str,dict_to_list(self.base_rec_parameters)))
        string="_" if len(list_parameters)>0 else ""
        return f"{self.city}_{self.base_rec}_{self.base_rec_list_size}"+\
                              string+'_'.join(list_parameters)

    def get_final_rec_name(self):
        list_parameters=list(map(str,dict_to_list(self.final_rec_parameters)))
        string="_" if len(list_parameters)>0 else ""
        return f"{self.get_base_rec_name()}_{self.final_rec}_{self.final_rec_list_size}"+\
            string+'_'.join(list_parameters)
    def get_base_rec_short_name(self):
        #list_parameters=list(map(str,dict_to_list(self.base_rec_parameters)))
        if self.base_rec == "mostpopular":
            return self.base_rec
        elif self.base_rec == "usg":
            return self.base_rec
        #string="_" if len(list_parameters)>0 else ""
    def get_final_rec_short_name(self):
        if self.final_rec == 'geocat':
            return f"{self.get_base_rec_short_name()}_{self.final_rec}"
        elif self.final_rec == 'persongeocat':
            return f"{self.get_base_rec_short_name()}_{self.final_rec}_{self.final_rec_parameters['cat_div_method']}"
        elif self.final_rec == 'geodiv':
            return f"{self.get_base_rec_short_name()}_{self.final_rec}"

    def get_base_rec_file_name(self):
        return self.get_base_rec_name()+".json"

    def get_final_rec_file_name(self):
        return self.get_final_rec_name()+".json"

    def load_base(self):
        CITY = self.city
        print(f"{CITY} city base loading")
        # Train load
        self.data_checkin_train = pickle.load(
            open(self.data_directory+"checkin/train/"+CITY+".pickle", "rb"))

        # Test load
        self.ground_truth = defaultdict(set)
        for checkin in pickle.load(open(self.data_directory+"checkin/test/"+CITY+".pickle", "rb")):
            self.ground_truth[checkin['user_id']].add(checkin['poi_id'])
        # Pois load
        self.poi_coos = {}
        self.poi_cats = {}
        for poi_id, poi in pickle.load(open(self.data_directory+"poi/"+CITY+".pickle", "rb")).items():
            self.poi_coos[poi_id] = tuple([poi['latitude'], poi['longitude']])
            self.poi_cats[poi_id] = poi['categories']

        # Social relations load
        self.social_relations = defaultdict(list)
        for user_id, friends in pickle.load(open(self.data_directory+"user/"+CITY+".pickle", "rb")).items():
            self.social_relations[user_id] = friends

        user_num = len(self.social_relations)
        poi_num = len(self.poi_coos)
        user_num, poi_num
        self.user_num=user_num
        self.poi_num=poi_num
        # Cat Hierarchy load
        self.dict_alias_title, self.category_tree, self.dict_alias_depth = cat_utils.cat_structs(
            self.data_directory+"categories.json")
        self.undirected_category_tree = self.category_tree.to_undirected()

        # Training matrix create
        self.training_matrix = np.zeros((user_num, poi_num))
        for checkin in self.data_checkin_train:
            self.training_matrix[checkin['user_id'], checkin['poi_id']] += 1

        # poi neighbors load
        self.poi_neighbors = pickle.load(
            open(self.data_directory+"neighbor/"+CITY+".pickle", "rb"))
        print(f"{CITY} city base loaded")
        self.all_uids = list(range(user_num))
        self.all_lids = list(range(poi_num))

    def run_usg(self, U, S, G, uid, alpha, beta):
        if uid in self.ground_truth:

            U_scores = normalize([U.predict(uid, lid)
                                  if self.training_matrix[uid, lid] == 0 else -1
                                  for lid in self.all_lids])
            S_scores = normalize([S.predict(uid, lid)
                                  if self.training_matrix[uid, lid] == 0 else -1
                                  for lid in self.all_lids])
            G_scores = normalize([G.predict(uid, lid)
                                  if self.training_matrix[uid, lid] == 0 else -1
                                  for lid in self.all_lids])

            U_scores = np.array(U_scores)
            S_scores = np.array(S_scores)
            G_scores = np.array(G_scores)

            overall_scores = (1.0 - alpha - beta) * U_scores + \
                alpha * S_scores + beta * G_scores

            predicted = list(reversed(overall_scores.argsort()))[
                :self.base_rec_list_size]
            overall_scores = list(reversed(np.sort(overall_scores)))[
                :self.base_rec_list_size]
            #actual = ground_truth[uid]
            # print(uid)
            return json.dumps({'user_id': uid, 'predicted': list(map(int, predicted)), 'score': list(map(float, overall_scores))})+"\n"
        return None

    def usg(self):
        training_matrix = self.training_matrix
        all_uids = self.all_uids
        social_relations = self.social_relations
        poi_coos = self.poi_coos
        top_k = self.base_rec_list_size
        training_matrix = training_matrix.copy()
        training_matrix[training_matrix >= 1] = 1
        alpha = 0.1
        beta = 0.1

        U = UserBasedCF()
        S = FriendBasedCF(eta=0.05)
        G = PowerLaw()

        U.pre_compute_rec_scores(training_matrix)
        S.compute_friend_sim(social_relations, training_matrix)
        G.fit_distance_distribution(training_matrix, poi_coos)

        executor = ProcessPoolExecutor()
        print("Running usg")
        futures = [executor.submit(
            self.run_usg, U, S, G, uid, alpha, beta) for uid in all_uids]
        results = [futures[i].result() for i in progressbar(range(len(futures)))]
        print("usg terminated")
        # dview.map_sync(run_usg,range(user_num))

        result_out = open(self.data_directory+"result/reclist/" + self.get_base_rec_file_name(), 'w')
        for json_string_result in results:
            result_out.write(json_string_result)
        result_out.close()
        print("Saved Rec List in "+self.data_directory+"result/reclist/" +
              self.city+"_sigir11_top_" + str(top_k) + ".json")

    def mostpopular(self):

        executor = ProcessPoolExecutor()

        futures = [executor.submit(self.run_mostpopular, uid)
                   for uid in self.all_uids]
        # results = [future.result() for future in futures]
        results = [futures[i].result() for i in progressbar(range(len(futures)))]

        result_out = open(self.data_directory+"result/reclist/"+self.get_base_rec_file_name(), 'w')
        for json_string_result in results:
            result_out.write(json_string_result)
        result_out.close()

    def run_mostpopular(self, uid):
        if uid in self.ground_truth:
            poi_indexes = set(list(range(self.poi_num)))
            visited_indexes = set(self.training_matrix[uid].nonzero()[0])
            not_visited_indexes = poi_indexes-visited_indexes
            not_visited_indexes = np.array(list(not_visited_indexes))
            poi_visits_nu = np.count_nonzero(self.training_matrix, axis=0)
            pois_score = poi_visits_nu/self.user_num
            for i in visited_indexes:
                pois_score[i] = 0
            predicted = list(reversed(np.argsort(pois_score)))[
                0:self.base_rec_list_size]
            overall_scores = list(reversed(np.sort(pois_score)))[
                0:self.base_rec_list_size]

            self.user_base_predicted_lid[uid]=predicted
            self.user_base_predicted_score[uid]=overall_scores
            return json.dumps({'user_id': uid, 'predicted': list(map(int, predicted)), 'score': list(map(float, overall_scores))})+"\n"
        return None

    def run_geocat(self, uid):
        if uid in self.ground_truth:
            predicted = self.user_base_predicted_lid[uid][
                0:self.base_rec_list_size]
            overall_scores = self.user_base_predicted_score[uid][
                0:self.base_rec_list_size]

            # start_time = time.time()

            predicted, overall_scores = gcobjfunc.geocat(uid, self.training_matrix, predicted, overall_scores,
                                                         self.poi_cats, self.poi_neighbors, self.final_rec_list_size, self.undirected_category_tree,
                                                         self.final_rec_parameters['div_geo_cat_weight'],self.final_rec_parameters['div_weight'])

            # print("uid → %d, time → %fs" % (uid, time.time()-start_time))

            predicted = np.array(predicted)[list(
                reversed(np.argsort(overall_scores)))]
            overall_scores = list(reversed(np.sort(overall_scores)))

            return json.dumps({'user_id': uid, 'predicted': list(map(int, predicted)), 'score': list(map(float, overall_scores))})+"\n"
        return None

    def geocat(self):
        executor = ProcessPoolExecutor()

        futures = [executor.submit(self.run_geocat, uid)
                for uid in self.all_uids]
        results = [futures[i].result() for i in progressbar(range(len(futures)))]
        result_out = open(self.data_directory+"result/reclist/"+self.get_final_rec_file_name(), 'w')
        for json_string_result in results:
            result_out.write(json_string_result)
        result_out.close()


    def run_persongeocat(self,uid):
        if uid in self.ground_truth:
            predicted = self.user_base_predicted_lid[uid][
                0:self.base_rec_list_size]
            overall_scores = self.user_base_predicted_score[uid][
                0:self.base_rec_list_size]
            
            # start_time = time.time()
            if self.geo_div_propensity[uid] == 0 and self.cat_div_propensity[uid] == 0:
                predicted, overall_scores = gcobjfunc.geocat(uid, self.training_matrix, predicted, overall_scores,
                                                self.poi_cats, self.poi_neighbors, self.final_rec_list_size, self.undirected_category_tree,
                                                0,0)
            else:
                div_geo_cat_weight=(self.geo_div_propensity[uid])/(self.geo_div_propensity[uid]+self.cat_div_propensity[uid])
                predicted, overall_scores = gcobjfunc.geocat(uid, self.training_matrix, predicted, overall_scores,
                                                         self.poi_cats, self.poi_neighbors, self.final_rec_list_size, self.undirected_category_tree,
                                                         div_geo_cat_weight,self.final_rec_parameters['div_weight'])

            # print("uid → %d, time → %fs" % (uid, time.time()-start_time))

            predicted = np.array(predicted)[list(
                reversed(np.argsort(overall_scores)))]
            overall_scores = list(reversed(np.sort(overall_scores)))

            return json.dumps({'user_id': uid, 'predicted': list(map(int, predicted)), 'score': list(map(float, overall_scores))})+"\n"
        return None

    def persongeocat(self):
        print("Computing geographic diversification propensity")
        pgeo_div_runner=GeoDivPropensity(self.training_matrix,self.poi_coos)
        self.geo_div_propensity= pgeo_div_runner.geo_div_walk()
        #print(geo_div_propensity)
        

        pcat_div_runner=CatDivPropensity(self.training_matrix,
        cat_utils.get_users_cat_visits(self.training_matrix,self.poi_cats),
        self.undirected_category_tree,cat_div_method=self.final_rec_parameters['cat_div_method'])
        print("Computing categoric diversification propensity")
        self.cat_div_propensity=pcat_div_runner.compute_cat_div_propensity()
       #print(self.cat_div_propensity)
        # self.beta=geo_div_propensity/np.add(geo_div_propensity,cat_div_propensity)
    

        executor = ProcessPoolExecutor()

        futures = [executor.submit(self.run_persongeocat, uid)
                for uid in self.all_uids]
        results = [futures[i].result() for i in progressbar(range(len(futures)))]
        result_out = open(self.data_directory+"result/reclist/"+self.get_final_rec_file_name(), 'w')
        for json_string_result in results:
            result_out.write(json_string_result)
        result_out.close() 


        # cat_div_propensity= pcat_div_runner

    def run_geodiv(self, uid):
        if uid in self.ground_truth:
            predicted = self.user_base_predicted_lid[uid][
                0:self.base_rec_list_size]
            overall_scores = self.user_base_predicted_score[uid][
                0:self.base_rec_list_size]
            
            # start_time = time.time()

            predicted, overall_scores = gcobjfunc.geodiv(uid, self.training_matrix, predicted, overall_scores,
                                                         self.poi_neighbors, self.final_rec_list_size,
                                                         self.final_rec_parameters['div_weight'])

            # print("uid → %d, time → %fs" % (uid, time.time()-start_time))

            predicted = np.array(predicted)[list(
                reversed(np.argsort(overall_scores)))]
            overall_scores = list(reversed(np.sort(overall_scores)))

            return json.dumps({'user_id': uid, 'predicted': list(map(int, predicted)), 'score': list(map(float, overall_scores))})+"\n"
        return None

    def geodiv(self):
        executor = ProcessPoolExecutor()

        futures = [executor.submit(self.run_geodiv, uid)
                for uid in self.all_uids]
        results = [futures[i].result() for i in progressbar(range(len(futures)))]
        result_out = open(self.data_directory+"result/reclist/"+self.get_final_rec_file_name(), 'w')
        for json_string_result in results:
            result_out.write(json_string_result)
        result_out.close()
    def load_base_predicted(self):
        result_file = open(self.data_directory+"result/reclist/"+self.get_base_rec_file_name(), 'r')
        for i,line in enumerate(result_file):
            obj=json.loads(line)
            self.user_base_predicted_lid[obj['user_id']]=obj['predicted']
            self.user_base_predicted_score[obj['user_id']]=obj['score']

    def load_final_predicted(self):
        result_file = open(self.data_directory+"result/reclist/"+self.get_final_rec_file_name(), 'r')

        for i,line in enumerate(result_file):
            obj=json.loads(line)
            self.user_final_predicted_lid[obj['user_id']]=obj['predicted']
            self.user_final_predicted_score[obj['user_id']]=obj['score']

    
    def run_base_recommender(self):
        base_recommender=self.BASE_RECOMMENDERS[self.base_rec]
        base_recommender()


    def run_final_recommender(self):
        final_recommender=self.FINAL_RECOMMENDERS[self.final_rec]
        if len(self.user_base_predicted_lid)>0:
            final_recommender()
        else:
            print("User base predicted list is empty")
        pass
    def run_all_base(self):
        for recommender in self.BASE_RECOMMENDERS:
            self.base_rec=recommender
            print(f"Running {recommender}")
            self.run_base_recommender()
    def run_all_final(self):
        print(f"Running all final recommenders, base recommender is {self.base_rec}")
        for recommender in self.FINAL_RECOMMENDERS:
            self.final_rec=recommender
            print(f"Running {recommender}")
            self.run_final_recommender()

    def eval(self,uid,*,base=False,k):
        if base:
            predicted=self.user_base_predicted_lid[uid]
        else:
            predicted=self.user_final_predicted_lid[uid]
        actual=self.ground_truth[uid]
        
        predicted_at_k=predicted[:k]
        precision_val=metrics.precisionk(actual, predicted_at_k)
        rec_val=metrics.recallk(actual, predicted_at_k)
        pr_val=metrics.prk(self.training_matrix[uid],predicted_at_k,self.poi_neighbors)
        ild_val=metrics.ildk(predicted_at_k,self.poi_cats,self.undirected_category_tree)
        gc_val=metrics.gck(uid,self.training_matrix,self.poi_cats,predicted_at_k)
        epc_val=metrics.epck(predicted_at_k,actual,uid,self.training_matrix)
        
        d={'user_id':uid,'precision':precision_val,'recall':rec_val,'pr':pr_val,'ild':ild_val,'gc':gc_val,'epc':epc_val}

        return json.dumps(d)+'\n'
    def eval_rec_metrics(self,*,base=False):
        for i,k in enumerate(experiment_constants.METRICS_K):
            print(f"running metrics at @{k}")
            
            if base:
                result_out = open(self.data_directory+"result/metrics/"+self.get_base_rec_name()+f"_{str(k)}{R_FORMAT}", 'w')
            else:
                result_out = open(self.data_directory+"result/metrics/"+self.get_final_rec_name()+f"_{str(k)}{R_FORMAT}", 'w')
            
            executor = ProcessPoolExecutor()
            futures = [executor.submit(self.eval, uid,base=base,k=k)
                    for uid in self.all_uids]
            results = [futures[i].result() for i in progressbar(range(len(futures)))]

            for json_string_result in results:
                result_out.write(json_string_result)
            result_out.close()

    def load_metrics(self,*,base=False):
        if base:
            rec_using=self.base_rec
            rec_short_name=self.get_base_rec_short_name()
        else:
            rec_using=self.final_rec
            rec_short_name=self.get_final_rec_short_name()
        
        self.metrics[rec_short_name]={}
        for i,k in enumerate(experiment_constants.METRICS_K):
            if base:
                result_file = open(self.data_directory+"result/metrics/"+self.get_base_rec_name()+f"_{str(k)}{R_FORMAT}", 'r')
            else:
                result_file = open(self.data_directory+"result/metrics/"+self.get_final_rec_name()+f"_{str(k)}{R_FORMAT}", 'r')
            
            self.metrics[rec_short_name][k]=[]
            for i,line in enumerate(result_file):
                obj=json.loads(line)
                self.metrics[rec_short_name][k].append(obj)




    def plot_bar_metrics(self):
        palette = plt.get_cmap('Set1')
        for i,k in enumerate(experiment_constants.METRICS_K):

            
            metrics_mean=dict()
            for i,rec_using,metrics in zip(range(len(self.metrics)),self.metrics.keys(),self.metrics.values()):
                metrics=metrics[k]
                
                metrics_mean[rec_using]=defaultdict(float)
                for obj in metrics:
                    for key in obj:
                        if key in self.metrics_name:
                            metrics_mean[rec_using][key]+=obj[key]
                
                
                for j,key in enumerate(metrics_mean[rec_using]):
                    metrics_mean[rec_using][key]/=len(metrics)
                    #print(f"{key}:{metrics_mean[rec_using][key]}")
            fig = plt.figure()
            ax=fig.add_subplot(111)
            barWidth= 1-len(self.metrics)/(1+len(self.metrics))
            N=len(self.metrics_name)
            indexes=np.arange(N)
            i=0
            for rec_using,rec_metrics in metrics_mean.items():
                print(f"{rec_using} at @{k}")
                print(rec_metrics.values())
                ax.bar(indexes+i*barWidth,rec_metrics.values(),barWidth,label=rec_using,color=palette(i))
                #ax.bar(indexes[j]+i*barWidth,np.mean(list(rec_metrics.values())),barWidth,label=rec_using,color=palette(i))
                i+=1
            metrics_utility_score=dict()
            for rec_using in metrics_mean:
                metrics_utility_score[rec_using]=dict()
            for metric_name in self.metrics_name:
                max_metric_value=0
                min_metric_value=1
                for rec_using,rec_metrics in metrics_mean.items():
                    max_metric_value=max(max_metric_value,rec_metrics[metric_name])
                    min_metric_value=min(min_metric_value,rec_metrics[metric_name])
                for rec_using,rec_metrics in metrics_mean.items():
                    metrics_utility_score[rec_using][metric_name]=(rec_metrics[metric_name]-min_metric_value)\
                        /(max_metric_value-min_metric_value)
            i=0
            for rec_using,utility_scores in metrics_utility_score.items():
                ax.bar(N+i*barWidth,np.sum(list(utility_scores.values()))/len(utility_scores),barWidth,label=rec_using,color=palette(i))
                i+=1
            
                

            #ax.set_xticks(np.arange(N+1)+barWidth*(np.floor((len(self.metrics))/2)-1)+barWidth/2)
            ax.set_xticks(np.arange(N+1)+barWidth*(((len(self.metrics))/2)-1)+barWidth/2)
            # ax.legend((p1[0], p2[0]), self.metrics_name)
            ax.legend(tuple(self.metrics.keys()))
            ax.set_xticklabels(self.metrics_name+['MAUT'])
            ax.set_title(f"at @{k}, {self.city}")
            fig.show()
            plt.show()
            fig.savefig(self.data_directory+f"result/img/all_met_{str(k)}.png")
                
                # ax.bar(indexes[j+1]+i*barWidth,np.mean(list(metrics_mean[rec_using].values())),barWidth,label=rec_using,color=palette(i))


    # def print_metrics(self):
        
    #     palette = plt.get_cmap('Set1')
    #     for i,k in enumerate(experiment_constants.METRICS_K):
    #         fig = plt.figure()
    #         ax=fig.add_subplot(111)
    #         #barWidth = 1-(len(self.metrics)-1)/len(self.metrics)
    #         barWidth= 0.25
    #         N=len(self.metrics_name)
    #         indexes=np.arange(N)
            
    #         for i,rec_using,metrics in zip(range(len(self.metrics)),self.metrics.keys(),self.metrics.values()):
    #             metrics=metrics[k]
                
    #             metrics_mean=defaultdict(float)
    #             for obj in metrics:
    #                 for key in obj:
    #                     metrics_mean[key]+=obj[key]
    #             print(f"Metrics at @{k}")
                
    #             del metrics_mean['user_id']
    #             for j,key in enumerate(metrics_mean):
    #                 metrics_mean[key]/=len(metrics)
    #                 print(f"{key}:{metrics_mean[key]}")
                    
    #                 ax.bar(indexes[j]+i*barWidth,metrics_mean[key],barWidth,label=rec_using,color=palette(i))
    #             ax.bar(indexes[j+1]+i*barWidth,np.mean(list(metrics_mean.values())),barWidth,label=rec_using,color=palette(i))
    #         ax.set_title(f"Metrics")
    #         ax.legend((p1[0], p2[0]), list(self.metrics.keys()))
    #         ax.set_xticks(indexes+barWidth/2)
    #         ax.set_xticklabels(metrics_name)
    #         fig.show()
    #         plt.show()
            #fig.savefig(self.data_directory+"result/img/"+self.get_final_rec_name()+f"_{str(k)}.png")
                
                
