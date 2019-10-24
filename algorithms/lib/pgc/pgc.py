import numpy as np
# import os
# import sys
# module_path = os.path.abspath(os.path.join('..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)

import lib.geo_utils as geo_utils
import collections


def cmean_dist_pois(poi_coos):
    lats=np.array([])
    longs=np.array([])
    for i,j in poi_coos.items():
        lats=np.append(lats,j[0])
        longs=np.append(longs,j[1])
    lat,lon = np.mean(lats),np.mean(longs)
    md=0
    for i in range(len(lats)):
        md+=geo_utils.haversine(lat,lon,lats[i],longs[i])
    return md/len(lats)

def cmean_dist_users(training_matrix,poi_coos):
    users_cmean=list()
    for i in range(training_matrix.shape[0]):
        lids=training_matrix[i].nonzero()[0]
        lats=np.array([])
        longs=np.array([])
        for lid in lids:
            lats=np.append(lats,poi_coos[lid][0])
            longs=np.append(longs,poi_coos[lid][1])
        lat,lon = np.mean(lats),np.mean(longs)
        md=0
        
        for i in range(len(lats)):
            md+=geo_utils.haversine(lat,lon,lats[i],longs[i])
        users_cmean.append(md/len(lats))
    return users_cmean

def cmedian_dist_pois(poi_coos):
    lats=np.array([])
    longs=np.array([])
    for i,j in poi_coos.items():
        lats=np.append(lats,j[0])
        longs=np.append(longs,j[1])
    lat,lon = np.median(lats),np.median(longs)
    md=0
    for i in range(len(lats)):
        md+=geo_utils.haversine(lat,lon,lats[i],longs[i])
    return md/len(lats)

def cmedian_dist_users(training_matrix,poi_coos):
    users_cmean=list()
    for i in range(training_matrix.shape[0]):
        lids=training_matrix[i].nonzero()[0]
        lats=np.array([])
        longs=np.array([])
        for lid in lids:
            lats=np.append(lats,poi_coos[lid][0])
            longs=np.append(longs,poi_coos[lid][1])
        lat,lon = np.median(lats),np.median(longs)
        md=0
        
        for i in range(len(lats)):
            md+=geo_utils.haversine(lat,lon,lats[i],longs[i])
        users_cmean.append(md/len(lats))
    return users_cmean


def geo_div_propensity(users_cmean_dist,cmean_dist_city):
    norm_prop=(users_cmean_dist/cmean_dist_city)
    norm_prop[norm_prop>1]=1
    return norm_prop


def cat_div_std_norm(cats_visits):
    std=np.std(cats_visits)
    if std == 0:
        return 0
    else:
        return std/(np.max(cats_visits)-np.min(cats_visits))


def cat_div_propensity(training_matrix,poi_cats):
    values=[]
    for i in range(training_matrix.shape[0]):
        cats_visits=collections.defaultdict(int)
        lids=training_matrix[i].nonzero()[0]
        for lid in lids:
            for cat in poi_cats[lid]:
                cats_visits[cat]+=training_matrix[i,lid]
        cv=np.array(list(cats_visits.values()))
        values.append(cat_div_std_norm(cv))
    return values





