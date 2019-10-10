import networkx as nx
def category_dis_sim(category1,category2,undirected_category_tree):
    dissim=0.0
    spd=nx.shortest_path_length(undirected_category_tree,category1,category2)
    sim = 1.0 / (1.0 + spd)
    dissim=1.0-sim
    return dissim

def min_dist_to_list_cat(poi_id,pois,poi_cats,undirected_category_tree):
    num_pois=len(pois)
    min_dissim=1.0
    cats_of_poi=poi_cats[poi_id]
    if num_pois==0:
        min_dissim=1.0
    else:
        for index_1 in pois:

            local_min_distance=1
            cur_distance=0
            for category1 in poi_cats[index_1]:
                for category2 in cats_of_poi:
                    cur_distance=category_dis_sim(
                        category1,
                        category2,undirected_category_tree)
                    local_min_distance=min(local_min_distance,cur_distance)
            min_dissim=min(min_dissim,local_min_distance)
    return min_dissim

def gc(poi_id,rec_list,relevant_cats,poi_cats):

    cats=set(poi_cats[poi_id])
    for lid in rec_list:
        cats.update(poi_cats[lid])
    count_equal=0
    for cat1 in relevant_cats:
        for cat2 in cats:
            if cat1 == cat2:
                #print(cat1)
                count_equal=count_equal+1    
    return count_equal/len(relevant_cats)

def update_geo_cov(poi_id,log_poi_ids,rec_list_size,poi_cover,poi_neighbors):
    log_size=len(log_poi_ids)
    
    #neighbors=[lid if lid in poi_neighbors[poi_id] for lid in user_pois.nonzero()[0]]
    neighbors=list()
    for id_neighbor in poi_neighbors[poi_id]:
        for i in range(log_size):
            log_poi_id=log_poi_ids[i]
            if log_poi_id == id_neighbor:
                neighbors.append(i)
    
        
    ###user_log_size=len(df_user_review)
    #neighbors=poi_neighbors(poi,df_user_review,0.5)
#     print("Start")
#     print(poi_neighbors)
#     print("End")
    ##neighbors=df_user_review[df_user_review['business_id'].isin(poi_neighbors.index.tolist())]

    num_neighbors=len(neighbors)
    #set_trace()
    vl=1
    COVER_OF_POI=log_size/rec_list_size
    accumulated_cover=0
    # Cover calc
    if num_neighbors<1:
        accumulated_cover+=COVER_OF_POI
    else:
        cover_of_neighbor=COVER_OF_POI/num_neighbors
        for index in neighbors:
            poi_cover[index]+=cover_of_neighbor
    accumulated_cover/=log_size
    # end PR and DP

    DP=0
    
    for i in range(log_size):
        lid=i
        if vl>=poi_cover[lid]:
            DP+=(vl-poi_cover[lid])**2
    DP+=(accumulated_cover**2)/2
    DP_IDEAL=log_size+0.5
    PR=1-DP/(DP_IDEAL)
    
    return PR


def ILD_GC_PR(score,ild_div,gc_div,pr,current_proportionality,rec_list_size,div_geo_cat_weight,div_weight):

    delta_proportionality=max(0,pr-current_proportionality)
    
    #delta_proportionality=max(0,update_geo_cov(poi,df_user_review,rec_list_size,business_cover.copy(),poi_neighbors)-current_proportionality)
    #print(poi.business_id,ild_div,gc_div,delta_proportionality)

    if delta_proportionality<0:
        delta_proportionality=0
    div_cat = gc_div+ild_div/rec_list_size
    div_geo = delta_proportionality
    div=div_geo_cat_weight*div_geo+(1-div_geo_cat_weight)*div_cat
    return (score**(1-div_weight))*(div**div_weight)
    
    
    