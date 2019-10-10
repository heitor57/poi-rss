import numpy as np
import pandas as pd

## Mostpopular rank by the top-k most visited pois by distinct users
def mostpopular(df_checkin,k,num_users,list_user_pois):
    rec_list=df_checkin[['business_id','user_id']].groupby('business_id').nunique()[['user_id']].sort_values('user_id',ascending=0).reset_index(level=0).rename(columns={"user_id":"score"})
    rec_list['score']=rec_list['score']/num_users
    rec_list=rec_list[rec_list.apply(lambda row: False if row['business_id'] in list_user_pois else True,axis=1)]
    return rec_list.iloc[0:k]


def mostpopularnp(training_matrix,uid):
    poi_indexes=set(list(range(training_matrix.shape[1])))
    visited_indexes=set(training_matrix[uid].nonzero()[0])
    not_visited_indexes=poi_indexes-visited_indexes
    not_visited_indexes=np.array(list(not_visited_indexes))
    poi_visits_nu=np.count_nonzero(training_matrix,axis=0)
    pois_score=poi_visits_nu/training_matrix.shape[0]
    for i in visited_indexes:
        pois_score[i]=0
#     indexes_arg_sort=list(reversed(np.argsort(pois_score)))



#     print(pois_score)
#     print(pois_score[indexes_arg_sort])
#     predicted=not_visited_indexes[indexes_arg_sort]
#     print(predicted)
    
    return pois_score