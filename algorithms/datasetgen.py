#!/usr/bin/env python
# coding: utf-8

# In[1]:


import lib.areamanager as areamanager
import pandas as pd
import json
import time
import collections
import numpy as np
import pickle
import lib.cat_utils as cat_utils
import lib.geo_utils as geo_utils
from lib.parallel_util import run_parallel
from lib.constants import geocat_constants,experiment_constants
from tqdm import tqdm
import math

SPLIT_YEAR=2017

# cities=['lasvegas','phoenix','charlotte','madison']
# cities=experiment_constants.CITIES
cities=['madison']

dict_alias_title,category_tree,dict_alias_depth=cat_utils.cat_structs("../data/categories.json")
undirected_category_tree=category_tree.to_undirected()
def category_filter(categories):
    tmp_cat_list=list()
    if categories != None:
        for category in categories:
            try:
                if dict_alias_depth[dict_alias_title[category]] <= 2:
                    tmp_cat_list.append(dict_alias_title[category])
            except:
                pass
        tmp_cat_list=cat_utils.get_most_detailed_categories(tmp_cat_list,dict_alias_title,dict_alias_depth)
    return tmp_cat_list

def category_normalization(categories):
    if categories != None:
        return categories
    else:
        return []

TRAIN_SIZE=experiment_constants.TRAIN_SIZE
TEST_SIZE=1-TRAIN_SIZE


# In[2]:


fbusiness=open("../data/business.json")
poi_data = dict()
start_time=time.time()
for i, line in enumerate(fbusiness):  
    # json to dict
    obj_json = json.loads(line)
    # add to the data collection
    if obj_json['categories'] != None:
        poi_data[obj_json['business_id']]={'latitude':obj_json['latitude'],
                         'longitude':obj_json['longitude'],
                         'categories':obj_json['categories'].split(', ')}
    else:
        poi_data[obj_json['business_id']]={'latitude':obj_json['latitude'],
                 'longitude':obj_json['longitude'],
                 'categories':obj_json['categories']}

print(time.time()-start_time)


# In[3]:


areas=dict()
for city in cities:
    areas[city]=areamanager.delimiter_area(city)


# In[4]:


cities_pid_in_area=dict()
start_time=time.time()
for city in cities:
    area=areas[city]
    pid_in_area=collections.defaultdict(bool)

    for poi_id in poi_data:
        if areamanager.poi_in_area(area,poi_data[poi_id]):

            pid_in_area[poi_id]=True

    cities_pid_in_area[city]=pid_in_area
print(time.time()-start_time)


# In[5]:


fuser=open("../data/user.json")
user_friend = dict()
user_data = dict()
start_time=time.time()
for i, line in enumerate(fuser):  
    # json to dict
    obj_json = json.loads(line)
    # add to the data collection
    user_friend[obj_json['user_id']]=obj_json['friends'].split(', ')
    custom_obj = dict()
    for key, value in obj_json.items():
        if key not in ['friends','elite','name','user_id']:
            custom_obj[key] = value

    user_data[obj_json['user_id']] = custom_obj

print(time.time()-start_time)


# In[6]:


freview=open("../data/review.json")

cities_checkin_data=dict()
for city in cities:
    cities_checkin_data[city]=list()

start_time=time.time()

for i, line in enumerate(freview):  
    # json to dict
    obj_json = json.loads(line)
    for city in cities:
        if cities_pid_in_area[city][obj_json['business_id']]:
            # add to the data collection
            cities_checkin_data[city].append({'user_id':obj_json['user_id'],
                             'poi_id':obj_json['business_id'],
                             'date':obj_json['date']})
            break
    if i % 500000 ==0:
        print(i)
print(time.time()-start_time)

ftip=open("../data/tip.json")
start_time=time.time()
for i, line in enumerate(ftip):  
    # json to dict
    obj_json = json.loads(line)
    for city in cities:
        if cities_pid_in_area[city][obj_json['business_id']]:
            # add to the data collection
            cities_checkin_data[city].append({'user_id':obj_json['user_id'],
                         'poi_id':obj_json['business_id'],
                         'date':obj_json['date']})
            break
    if i % 500000 ==0:
        print(i)
print(time.time()-start_time)


# In[ ]:


# df_checkin=pd.read_csv("../data/checkin.csv")

# df_checkin=df_checkin.set_index("user_id")


# In[ ]:


# city_area=areamanager.delimiter_area('madison')
# df_checkin_city=areamanager.pois_in_area(city_area,df_checkin.reset_index())


# In[ ]:


# i=0
# for idx,checkin in df_checkin.iterrows():
#    # print(checkin.business_id)
#     if cities_pid_in_area['madison'][checkin.business_id]:
#         i+=1

# i


# In[ ]:


# print(len(df_checkin_city['business_id'].drop_duplicates()))
# print(len(df_checkin_city['user_id'].drop_duplicates()))
# print(len(df_checkin_city))


# In[7]:


genoptions=['poi','neighbor','user','checkin'# ,'test','train'
            ,'user_data']
genoptions=['checkin',
            'poi','neighbor',
            'user','user_data'
            ]


# In[ ]:




for city in cities:
    print("CITY: %s" % (city))
    # Pega os checkins da cidade
    checkin_data=cities_checkin_data[city]
    print("checkin_data size: %d"%(len(checkin_data)))
    # transforma em dataframe
    df_checkin=pd.DataFrame.from_dict(checkin_data)
    df_checkin.head(1)

    # Começa a parte de filtragrem
    df_diff_users_visited=df_checkin[['user_id','poi_id']].drop_duplicates().reset_index(drop=True).groupby('poi_id').count().reset_index().rename(columns={"user_id":"diffusersvisited"})

    df_diff_users_visited=df_diff_users_visited[df_diff_users_visited['diffusersvisited']>=5]

    del df_diff_users_visited['diffusersvisited']
    df_checkin=pd.merge(df_checkin,df_diff_users_visited,on='poi_id',how='inner')
    df_checkin['Count']=df_checkin.groupby(['user_id'])['user_id'].transform('count')
    df_checkin=df_checkin[df_checkin['Count']>=20]
    del df_checkin['Count']
    # converte para dicionario, ou lista de dicionarios
    checkin_data=list(df_checkin.to_dict('index').values())

    # termina a parte de filtragem
    
    # pega todos ids dos usuarios
    users_id = set()
    for check in checkin_data:
        users_id.add(check['user_id'])
    
    # quantidade de usuarios
    user_num=len(users_id)

    # pega todos ids dos pois
    pois_id = set()
    for check in checkin_data:
        pois_id.add(check['poi_id'])
   
    #quantidade de pois
    poi_num=len(pois_id)
    print("user_num:%d, poi_num:%d"%(user_num,poi_num))

    
    # Começa a transformar ids de String para inteiro
    users_id_to_int = dict()
    for i,user_id in enumerate(users_id):
        users_id_to_int[user_id]=i

    fuid=open('../data/user/id/'+city+'.pickle','wb')
    pickle.dump(users_id_to_int,fuid)
    fuid.close()

    pois_id_to_int = dict()
    
    for i,poi_id in enumerate(pois_id):
        pois_id_to_int[poi_id]=i

    fpid=open('../data/poi/id/'+city+'.pickle','wb')
    pickle.dump(pois_id_to_int,fpid)
    fpid.close()

    # Termina de transformar ids de String para inteiro
        
    # cria dicionario de "objetos" ou dicionarios de pois da cidade
    # alem de aplicar filtragem categorica

    city_poi_data=dict()
    if 'poi' in genoptions:
        for poi_id in pois_id:
            city_poi_data[pois_id_to_int[poi_id]]=poi_data[poi_id].copy()
            city_poi_data[pois_id_to_int[poi_id]] = {'categories':category_normalization(city_poi_data[pois_id_to_int[poi_id]]['categories'])}
        fpoi=open('../data/poi_full/'+city+'.pickle','wb')
        pickle.dump(city_poi_data,fpoi)
        fpoi.close()

    city_poi_data=dict()
    if 'poi' in genoptions:
        for poi_id in pois_id:
            city_poi_data[pois_id_to_int[poi_id]]=poi_data[poi_id].copy()
            city_poi_data[pois_id_to_int[poi_id]]['categories']=category_filter(poi_data[poi_id]['categories'])
        fpoi=open('../data/poi/'+city+'.pickle','wb')
        pickle.dump(city_poi_data,fpoi)
        fpoi.close()

    # pega os vizinhos de cada poi
#     print("Pegando vizinhos...")
    if 'neighbor' in genoptions:
        poi_neighbors={}
        pois_id=[pois_id_to_int[pid] for pid in pois_id]
        pois_coos = np.array([(city_poi_data[pid]['latitude'],city_poi_data[pid]['longitude']) for pid in pois_id])
        
        poi_coos_balltree = sklearn.neighbors.BallTree(poi_coos,metric="haversine")

        poi_neighbors = {lid: self.poi_coos_balltree.query_radius([pois_coos[lid]],NEIGHBOR_DISTANCE) for lid in pois_id}
        # args=[(lid,) for lid in pois_id]
        # def neighbors_searcher(poi_id):
        #     neighbors=list()
        #     for npoi_id in pois_id:
        #         if geo_utils.dist((city_poi_data[poi_id]['latitude'],city_poi_data[poi_id]['longitude']),(city_poi_data[npoi_id]['latitude'],city_poi_data[npoi_id]['longitude'])) <= geocat_constants.NEIGHBOR_DISTANCE:
        #             neighbors.append(npoi_id)
        #     return neighbors
        # poi_neighbors = run_parallel(neighbors_searcher,args,chunksize=60)
        # list to dict
        # poi_neighbors = {i: poi_neighbors[i] for i in range(len(poi_neighbors))}
        print("Terminou vizinhos...")
        fneighbors=open('../data/neighbor/'+city+'.pickle','wb')
        pickle.dump(poi_neighbors,fneighbors)
        fneighbors.close()
    
    city_user_friend=dict()
    countusf=0
    print("Inicio Amigos...")
    users_id = list(users_id)
    if 'user' in genoptions:
        for i in tqdm(range(len(users_id))):
            user_id=users_id[i]
            ucity_friends=list()
            for friend_id in user_friend[user_id]:
                try:
                    ucity_friends.append(users_id_to_int[friend_id])
                    countusf+=1
                except:
                    pass

            city_user_friend[users_id_to_int[user_id]]=ucity_friends
        fuser=open('../data/user/friend/'+city+'.pickle','wb')
        pickle.dump(city_user_friend,fuser)
        fuser.close()
    
    print("Fim Amigos...")
    print("Friends: %d"%(countusf))

    city_user_data = dict()
    if 'user_data' in genoptions:
        for i in tqdm(range(len(users_id))):
            user_id=users_id[i]
            city_user_data[users_id_to_int[user_id]]=user_data[user_id].copy()
        fuser=open('../data/user/'+city+'.pickle','wb')
        pickle.dump(city_user_data,fuser)
        fuser.close()

    if 'checkin' in genoptions:
        for checkin in checkin_data:
            checkin['user_id'] = users_id_to_int[checkin['user_id']]
            checkin['poi_id'] = pois_id_to_int[checkin['poi_id']]
            checkin['date'] = pd.to_datetime(checkin['date'])
        fcheckin=open('../data/checkin/'+city+'.pickle','wb')
        pickle.dump(checkin_data,fcheckin)
        fcheckin.close()
    #### Treino e teste por ano
    # -==============================ANO=========================================.....
#     df_test_checkin=pd.DataFrame(checkin_data)
#     df_test_checkin=df_test_checkin[df_test_checkin.date>=pd.to_datetime("01/01/2017")].reset_index(drop=True)
#     #print(pd.DataFrame.from_dict(checkin))
#     df_train_checkin=pd.DataFrame(checkin_data)
#     df_train_checkin=df_train_checkin[df_train_checkin.date<pd.to_datetime("01/01/2017")].reset_index(drop=True)
    
#     te_checkin_data=list(df_test_checkin.to_dict('index').values())
#     tr_checkin_data=list(df_train_checkin.to_dict('index').values())
    # -==============================ANO=========================================.....
    
    
    #### Treino e teste com porcentagem
    # -==============================PORCENTAGEM====================================================.....
    tr_checkin_data=[]
    te_checkin_data=[]
    
    user_checkin_data =dict()
    for user_id in users_id:
        user_checkin_data[users_id_to_int[user_id]]=list()
    
    for checkin in checkin_data:
        user_checkin_data[checkin['user_id']].append({'poi_id':checkin['poi_id'],'date':checkin['date']})
    
    for i in tqdm(range(len(users_id))):
        user_id=users_id_to_int[users_id[i]]
        checkin_list=user_checkin_data[user_id]
        checkin_list=sorted(checkin_list, key = lambda i: i['date']) 
        train_size=math.ceil(len(checkin_list)*TRAIN_SIZE)
        #test_size=math.floor(len(checkin_list)*TEST_SIZE)
        count=1
        te_pois=set()
        tr_pois=set()
        initial_te_size=len(te_checkin_data)
        final_te_size=len(te_checkin_data)
        for checkin in checkin_list:
            if count<=train_size:
                tr_pois.add(checkin['poi_id'])
                tr_checkin_data.append({'user_id':user_id,'poi_id':checkin['poi_id'],'date':checkin['date']})
            else:
                te_pois.add(checkin['poi_id'])
                te_checkin_data.append({'user_id':user_id,'poi_id':checkin['poi_id'],'date':checkin['date']})
                final_te_size+=1
            count+=1
        int_pois=te_pois&tr_pois
        rel_index=0
        for i in range(initial_te_size,final_te_size):
            i+=rel_index
            if te_checkin_data[i]['poi_id'] in int_pois:
                te_checkin_data.pop(i)
                rel_index-=1
    # -==============================PORCENTAGEM====================================================.....
    #### Treino e teste com porcentagem


    ftecheckin=open('../data/checkin/test/'+city+'.pickle','wb')
    pickle.dump(te_checkin_data,ftecheckin)
    ftecheckin.close()
    ftrcheckin=open('../data/checkin/train/'+city+'.pickle','wb')
    pickle.dump(tr_checkin_data,ftrcheckin)
    ftrcheckin.close()
    


# In[ ]:


# pd.read_csv('../data/user/madison.csv')


# In[ ]:


# charl=pickle.load(open('../data/user/charlotte.pickle','rb'))

# a=0
# for i in charl:
    
#     a+=len(charl[i])
# a


# In[ ]:


# df_checkin=pd.DataFrame.from_dict(checkin_data)
# df_checkin.head(1)


# In[ ]:


# len(checkin_data)

# users_id = set()
# for check in checkin_data:
#     users_id.add(check['user_id'])
# #users_id=list(users_id)

# user_num=len(users_id)
# user_num

# pois_id = set()
# for check in checkin_data:
#     pois_id.add(check['poi_id'])
# #pois_id=list(pois_id)

# poi_num=len(pois_id)
# poi_num

# users_id_to_int = dict()
# for i,user_id in enumerate(users_id):
#     users_id_to_int[user_id]=i

# pois_id_to_int = dict()
# for i,poi_id in enumerate(pois_id):
#     pois_id_to_int[poi_id]=i

# training_matrix = np.zeros((len(users_id),len(pois_id)))
# for check in checkin_data:
#     training_matrix[users_id_to_int[check['user_id']],pois_id_to_int[check['poi_id']]]+=1

# diff_visits=np.count_nonzero(training_matrix,axis=0)

# lids_subset=np.nonzero(diff_visits>=5)[0]

# training_matrix=training_matrix[:,lids_subset]

# users_visits=np.sum(training_matrix,axis=1)

# uids_subset=np.nonzero(users_visits>=20)[0]

# training_matrix=training_matrix[uids_subset,:]
# np.sum(training_matrix)

