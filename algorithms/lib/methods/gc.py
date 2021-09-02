from .objfunc import gc

def gc_objective_function(poi_id, score, rec_list, relevant_cats, poi_cats, div_weight):
    div = gc(poi_id,rec_list,relevant_cats,poi_cats)
    return (score**(1-div_weight))*(div**div_weight)

def gc_diversifier(uid,training_matrix,tmp_rec_list,tmp_score_list,
                   poi_cats,undirected_category_tree,
                   div_weight, K):
    range_K=range(K)
    rec_list=[]

    lids=training_matrix[uid].nonzero()[0]

    # lid_visits=training_matrix[:,lids].sum(axis=0)
    lid_visits=training_matrix[uid,lids]#.sum(axis=0)
    mean_visits=lid_visits.mean()
    relevant_lids=lids[lid_visits>mean_visits]
    relevant_cats=set()
    for lid in relevant_lids:
        relevant_cats.update(poi_cats[lid])

    final_scores=[]
    for i in range_K:
        poi_to_insert=None
        max_objective_value=-200
        for j in range(len(tmp_rec_list)):
            candidate_poi_id=tmp_rec_list[j]
            candidate_score=tmp_score_list[j]
            objective_value=gc_objective_function(candidate_poi_id, candidate_score,
                                                  rec_list, relevant_cats,
                                                  poi_cats, div_weight)
            
            if objective_value > max_objective_value:
                max_objective_value=objective_value
                poi_to_insert=candidate_poi_id
        if poi_to_insert is not None:
            rm_idx=tmp_rec_list.index(poi_to_insert)
            tmp_rec_list.pop(rm_idx)
            tmp_score_list.pop(rm_idx)
            rec_list.append(poi_to_insert)
            final_scores.append(max_objective_value)

    return rec_list,final_scores
