import numpy as np
import networkx as nx
import methods.objfunc
import geo_utils
from methods.objfunc import category_dis_sim

np.seterr(all='raise')

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

def f1k(precision, recall):
    if precision == 0 and recall == 0:
        return 0
    return 2*(precision*recall)/(precision+recall)
    

def ndcgk(actual, predicted, k):
    idcg = 1.0
    dcg = 1.0 if predicted[0] in actual else 0.0
    for i,p in enumerate(predicted[1:]):
        if p in actual:
            dcg += 1.0 / np.log(i+2)
        idcg += 1.0 / np.log(i+2)
    return dcg / idcg

def ildk(pois,poi_cats,undirected_category_tree):
    
    # min_dissim=1.0
    num_pois=len(pois)
    local_ild=0
    count=0
 
    if num_pois == 0 or num_pois == 1:
        print("Number of pois:",num_pois)
        return 1.0
    else:
        for i, poi_1 in enumerate(pois):
            for j, poi_2 in enumerate(pois):
                if j < i:
                    local_min_distance=1
                    cur_distance=0
                    for category1 in poi_cats[poi_1]:
                        for category2 in poi_cats[poi_2]:
                            cur_distance=category_dis_sim(
                                category1,
                                category2,undirected_category_tree)
                            #print(category1,category2,cur_distance,local_min_distance)
                            local_min_distance=min(local_min_distance,cur_distance)
                    #min_dissim=min(min_dissim,local_min_distance)
                    local_ild+=local_min_distance
                    count+=1

    return local_ild/count

def ildgk(pois,poi_coos):
    local_ild_km=0
    count=0
    num_pois=len(pois)

    if num_pois == 0 or num_pois == 1:
        print("Number of pois:",num_pois)
        return 0
    else:
        for lid1 in pois:
            lat1, lon1 = poi_coos[lid1]
            for lid2 in pois:
                if lid1 != lid2:
                    lat2, lon2 = poi_coos[lid2]
                    local_ild_km += geo_utils.haversine(lat1,lon1,lat2,lon2)
                    count+=1
    return local_ild_km/count

def gck(uid,training_matrix,poi_cats,predicted):
    lids=training_matrix[uid].nonzero()[0]
    # lid_visits=training_matrix[:,lids].sum(axis=0)
    lid_visits=training_matrix[uid,lids]#.sum(axis=0)
    mean_visits=lid_visits.mean()
    relevant_lids=lids[lid_visits>mean_visits]

    relevant_cats=set()
    for lid in relevant_lids:
        relevant_cats.update(poi_cats[lid])
    if len(relevant_cats) == 0:
        return 0
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


def old_epck(rec_list,actual,pop,epc_numerator,epc_denominator):
    epc=0
    local_epc_numerator=0
    local_epc_denominator=0
    for i,lid in enumerate(rec_list):
        if lid in actual:
            tmpval=np.log(i+2)/np.log(2)
            local_epc_numerator+=(1-pop[lid])/tmpval
            local_epc_denominator+=1/tmpval
    epc_numerator.append(local_epc_numerator)
    epc_denominator.append(local_epc_denominator)

def old_global_epck(training_matrix,ground_truth,predictions,all_uids,k):
    pop = epc_pop_list(training_matrix)
    epc_numerator = []
    epc_denominator = []
    for uid in all_uids:
        old_epck(predictions[uid][:k],ground_truth[uid],pop,epc_numerator,epc_denominator)
    return np.sum(epc_numerator)/np.sum(epc_denominator)

def epck(rec_list,actual,uid,training_matrix):
# ####    print(f"RecListSize:{len(rec_list)}")
#     C = 1/len(rec_list)

    C_2 = 1.0/len(rec_list)
#     sum_1=0
    sum_2=0
# ####    print("users:",training_matrix.shape[0])
    for i,lid in enumerate(rec_list):
        if lid in actual:
#             discount_k=pow(0.85,i)
#             prob_rel_k=1
            prob_seen_k=np.count_nonzero(training_matrix[:,lid])/training_matrix.shape[0]
#             sum_1+=discount_k*prob_rel_k*(1-prob_seen_k)
            sum_2 += 1-prob_seen_k
#     EPC_r = C*sum_1
    EPC=C_2*sum_2
    return EPC

def divgeocatk(ild_value,gc_value,pr_value,div_geo_cat_weight,K,div_cat_weight):
    div_cat = (1-div_cat_weight)*gc_value+div_cat_weight*ild_value
    div_geo = pr_value
    div=(1-div_geo_cat_weight)*div_geo+(div_geo_cat_weight)*div_cat
    return div

def relk(score_list, size):
    relevance = 0
    for i in score_list:
        relevance += i
    relevance /= len(score_list)
    return relevance

def calculate_fo(current_solution, poi_cats, undirected_category_tree, user_log,
                 poi_neighbors, div_geo_cat_weight, div_weight, K, relevant_cats, div_cat_weight):

    diversity = divgeocatk(
        ildk(current_solution.item_list, poi_cats, undirected_category_tree),
        geocat.objfunc.gc_list(current_solution.item_list,relevant_cats,poi_cats), #Cobertura de gêneros
        prk(user_log, current_solution.item_list, poi_neighbors),
        div_geo_cat_weight,
        K,
        div_cat_weight,
    )

    relevance = relk(current_solution.score_list, K)

    current_solution.diversity = diversity
    current_solution.relevance = relevance

    return (relevance**(1-div_weight))*(diversity**div_weight)

def pso_calculate_fo(current_particle, poi_cats, undirected_category_tree, user_log,
                     poi_neighbors, div_geo_cat_weight, div_weight, K, relevant_cats, dbest,div_cat_weight):

    diversity = divgeocatk(
        ildk(current_particle.item_list, poi_cats, undirected_category_tree),
        geocat.objfunc.gc_list(current_particle.item_list,relevant_cats,poi_cats), #Cobertura de gêneros
        prk(user_log, current_particle.item_list, poi_neighbors),
        div_geo_cat_weight,
        K,
        div_cat_weight,
    )

    relevance = relk(current_particle.score_list, K)

    fo = (relevance**(1-div_weight))*(diversity**div_weight)

    if current_particle.best_fo < fo:
        current_particle.best_fo = fo
        current_particle.best_relevance = relevance
        current_particle.best_diversity = diversity
        current_particle.best_item_list.clear()
        current_particle.best_score_list.clear()
        current_particle.best_item_list = current_particle.item_list.copy()
        current_particle.best_score_list = current_particle.score_list.copy()

    # Update div best
    if diversity > dbest.diversity:
        dbest.diversity = diversity
        dbest.fo = fo
        dbest.relevance = relevance
        dbest.item_list.clear()
        dbest.score_list.clear()
        dbest.item_list = current_particle.item_list.copy()
        dbest.score_list = current_particle.score_list.copy()
