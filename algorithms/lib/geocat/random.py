import random

def random_objective_function(score,div_weight):
    div = random.uniform(0,1)
    return (score**(1-div_weight))*(div**div_weight)
def random_diversifier(tmp_rec_list, tmp_score_list, K, div_weight):
    rec_list=[]
    final_scores=[]
    for i in range(K):
        poi_to_insert=None
        max_objective_value=-200
        for j in range(len(tmp_rec_list)):
            candidate_poi_id=tmp_rec_list[j]
            candidate_score=tmp_score_list[j]
            objective_value=random_objective_function(candidate_score,div_weight)
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
