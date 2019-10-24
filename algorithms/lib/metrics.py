import numpy as np
import networkx as nx

def mapk(actual, predicted, k):
    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def precisionk(actual, predicted):
    return 1.0 * len(set(actual) & set(predicted)) / len(predicted)


def recallk(actual, predicted):
    return 1.0 * len(set(actual) & set(predicted)) / len(actual)


def ndcgk(actual, predicted, k):
    idcg = 1.0
    dcg = 1.0 if predicted[0] in actual else 0.0
    for i,p in enumerate(predicted[1:]):
        if p in actual:
            dcg += 1.0 / np.log(i+2)
        idcg += 1.0 / np.log(i+2)
    return dcg / idcg


def category_dis_sim(category1,category2,undirected_category_tree):
    dissim=0.0
    spd=nx.shortest_path_length(undirected_category_tree,category1,category2)
    sim = 1.0 / (1.0 + spd)
    dissim=1.0-sim
    return dissim
    

def ildk(pois,poi_cats,undirected_category_tree):
    
    min_dissim=1.0
    num_pois=len(pois)
    local_ild=0
    local_ild_km=0
    count=0
    
    if num_pois==0:
        min_dissim=1.0
    else:
        for index_1 in pois:
            for index_2 in pois:
                if index_1 != index_2:
                    local_min_distance=1
                    cur_distance=0
                    for category1 in poi_cats[index_1]:
                        for category2 in poi_cats[index_2]:
                            cur_distance=category_dis_sim(
                                category1,
                                category2,undirected_category_tree)
                            #print(category1,category2,cur_distance,local_min_distance)
                            local_min_distance=min(local_min_distance,cur_distance)
                    #min_dissim=min(min_dissim,local_min_distance)
                    local_ild+=local_min_distance
                    count+=1
    
    return local_ild/count

def gck(uid,training_matrix,poi_cats,predicted):
    lids=training_matrix[uid].nonzero()[0]
    lid_visits=training_matrix[:,lids].sum(axis=0)
    mean_visits=lid_visits.mean()
    relevant_lids=lids[lid_visits>mean_visits]

    relevant_cats=set()
    for lid in relevant_lids:
        relevant_cats.update(poi_cats[lid])
    predicted_cats=set()
    for lid in predicted:
        predicted_cats.update(poi_cats[lid])
    count_equal=0
    for cat1 in relevant_cats:
        for cat2 in predicted_cats:
            if cat1 == cat2:
                #print(cat1)
                count_equal=count_equal+1    
    return count_equal/len(relevant_cats)

def prk(user_log,rec_list,poi_neighbors):
    
    log_poi_ids=list()
    poi_cover=list()
    for lid in user_log.nonzero()[0]:
        for visits in range(int(user_log[lid])):
            poi_cover.append(0)
            log_poi_ids.append(lid)
            
    log_size=len(log_poi_ids)
    rec_list_size=len(rec_list)
    COVER_OF_POI=log_size/rec_list_size
    vl=1
    accumulated_cover=0

    for poi_id in rec_list:
        
        neighbors=list()
        for id_neighbor in poi_neighbors[poi_id]:
            for i in range(log_size):
                log_poi_id=log_poi_ids[i]
                if log_poi_id == id_neighbor:
                    neighbors.append(i)
        num_neighbors=len(neighbors)
        #set_trace()
        
        
        
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

def epc_pop_list(training_matrix):
    visits=training_matrix.sum(axis=0)
    return visits/np.max(visits)


def epck(rec_list,actual,uid,pop,epc_numerator,epc_denominator):
    epc=0
    local_epc_numerator=0
    local_epc_denominator=0
    for i,lid in enumerate(rec_list):
        if lid in actual:
            tmpval=np.log(i+2)/np.log(2)
            local_epc_numerator+=np.log(1-pop[lid])/tmpval
            local_epc_denominator+=1/tmpval
    epc_numerator.append(local_epc_numerator)
    epc_denominator.append(local_epc_denominator)
    
    return epc_numerator.sum()/epc_denominator.sum()


