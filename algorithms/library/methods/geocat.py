### ILD
import numpy as np
import pandas as pd
import networkx as nx
import areamanager
import math
import geo_utils
import timeit
NEIGHBOR_DISTANCE=0.5# km
from IPython.core.debugger import set_trace
def category_in_rec_list(rec_list):
    categories_in_rec_list=set()
    
    for index, row in rec_list.iterrows():
        for category in row['categories']:
            categories_in_rec_list.add(category)
    
#     for category in poi['categories']:
#         categories_in_rec_list.add(category)
    return categories_in_rec_list

def category_dis_sim(category1,category2,undirected_category_tree):
    dissim=0.0
    spd=nx.shortest_path_length(undirected_category_tree,category1,category2)
    sim = 1.0 / (1.0 + spd)
    dissim=1.0-sim
    return dissim
    

def min_dist_to_list_cat(business,rec_list,dict_alias_title,undirected_category_tree):
    
    min_dissim=1.0
    rec_list_size=len(rec_list)
    if rec_list_size==0:
        min_dissim=1.0
    else:
        for index,row in rec_list.iterrows():
            local_min_distance=1
            cur_distance=0
            for category1 in business['categories']:
                for category2 in row['categories']:
                    cur_distance=category_dis_sim(
                        category1,
                        category2,undirected_category_tree)
                    #print(category1,category2,cur_distance,local_min_distance)
                    local_min_distance=min(local_min_distance,cur_distance)
            min_dissim=min(min_dissim,local_min_distance)
    
    return min_dissim
        
def objective_ild(business,rec_list,dict_alias_title,undirected_category_tree):
    # /*(1-_cfg._divWeight) * relevance + */_cfg._divWeight * diversity; where _cfg._divWeight = 0.9
    ret_val= 0.0
    diversity=0.0
    diversity=min_dist_to_list_cat(business,rec_list,dict_alias_title,undirected_category_tree)
    ret_val = diversity #+ (1-div_weight)*relevance
    return ret_val

### Genre Coverage


def relevant_categories_to_the_user(df_user_checkin):
    
    df_num_checkins=df_user_checkin.groupby("business_id").count()['date']

    mean_poi_visits=df_num_checkins.mean()

    df_num_checkins=pd.merge(df_user_checkin.reset_index(drop=True).drop_duplicates(subset=['business_id']),df_num_checkins,on='business_id')
    print(mean_poi_visits)
    # Relevant categories
    relevant_categories = set()    
    for index,checkin in df_num_checkins.iterrows():
        # Check if poi is relevant
        if checkin['date_y'] >= mean_poi_visits:
            
            # add relevant categories
            for category in checkin['categories']:
                relevant_categories.add(category)
    return relevant_categories
    

def objective_genre_coverage(poi,rec_list,df_user_review,relevant_categories,rec_list_categories):
    
    count_equal=0
#     categories_in_rec_list=set()
    
#     for index, row in rec_list.iterrows():
#         for category in row['categories']:
#             categories_in_rec_list.add(category)
    
#     for category in poi['categories']:
#         categories_in_rec_list.add(category)
    
    #print(categories_in_rec_list)
    categories=set()
    for index,poi_categories in rec_list['categories'].iteritems():
        categories.update(poi_categories)
    categories.update(poi['categories'])
    for cat1 in relevant_categories:
        for cat2 in categories:
            if cat1 == cat2:
                #print(cat1)
                count_equal=count_equal+1
    
    return count_equal/len(relevant_categories)

### PR


def poi_distance_from_pois(poi,df_poi):
#     result=np.array([])
#     for index,p in df_poi.iterrows():
#         result=np.append(result,geo_utils.mercator(poi['latitude'],poi['longitude'],p['latitude'],p['longitude']))
#     return result
    return geo_utils.haversine(poi['latitude'],poi['longitude'],df_poi['latitude'],df_poi['longitude'])

def poi_neighbors(poi,df_poi,distance_km):
    return df_poi[poi_distance_from_pois(poi,df_poi)<=distance_km]

def update_geo_cov(poi,df_user_review,rec_list_size,business_cover,poi_neighbors):
    
    user_log_size=len(df_user_review)
    #neighbors=poi_neighbors(poi,df_user_review,0.5)
#     print("Start")
#     print(poi_neighbors)
#     print("End")
    neighbors=df_user_review[df_user_review['business_id'].isin(poi_neighbors.index.tolist())]

    num_neighbors=len(neighbors)
    #set_trace()
    vl=1
    COVER_OF_POI=user_log_size/rec_list_size
    accumulated_cover=0
    # Cover calc
    if num_neighbors<1:
        accumulated_cover+=COVER_OF_POI
    else:
        cover_of_neighbor=COVER_OF_POI/num_neighbors
        for index,poi in neighbors.iterrows():
            business_cover[index]+=cover_of_neighbor
    accumulated_cover/=user_log_size
    # end PR and DP

    DP=0
    
    for index,business in df_user_review.iterrows():
        if vl>=business_cover[index]:
            DP+=(vl-business_cover[index])**2
    DP+=(accumulated_cover**2)/2
    DP_IDEAL=user_log_size+0.5
    PR=1-DP/(DP_IDEAL)
    
    return PR

### Geo-Cat
def objective_ILD_GC_PR(poi,df_user_review,rec_list,rec_list_size,business_cover,current_proportionality,div_geo_cat_weight,div_weight,dict_alias_title,undirected_category_tree,relevant_categories_user,rec_list_categories,poi_neighbors):
#     start = timeit.default_timer()
    ild_div=objective_ild(poi,rec_list,dict_alias_title,undirected_category_tree)

    #print(ild_div)
    gc_div=0
#     stop = timeit.default_timer()
#     print('Timea:', stop - start)
#     start = timeit.default_timer()
    gc_div=objective_genre_coverage(poi,rec_list,df_user_review,relevant_categories_user,rec_list_categories)

#     stop = timeit.default_timer()
#     print('Timeb:', stop - start)
#     start = timeit.default_timer()
    delta_proportionality=max(0,update_geo_cov(poi,df_user_review,rec_list_size,business_cover.copy(),poi_neighbors)-current_proportionality)
    print(poi.business_id,ild_div,gc_div,delta_proportionality)
    ##print(gc_div,delta_proportionality)
#     stop = timeit.default_timer()
#     print('Timec:', stop - start)
   # set_trace()
    if delta_proportionality<0:
        delta_proportionality=0
    div_cat = gc_div+ild_div/rec_list_size
    div_geo = delta_proportionality
    div=div_geo_cat_weight*div_geo+(1-div_geo_cat_weight)*div_cat
    return (poi['score']**(1-div_weight))*(div**div_weight)

