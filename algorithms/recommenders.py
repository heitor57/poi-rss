import numpy as np
import pandas as pd

## Mostpopular rank by the top-k most visited pois by distinct users
def mostpopular(df_checkin,k,df_user_checkin,num_users,list_user_pois):
    rec_list=df_checkin[['business_id','user_id']].groupby('business_id').nunique()[['user_id']].sort_values('user_id',ascending=0).reset_index(level=0).rename(columns={"user_id":"score"})
    rec_list['score']=rec_list['score']/num_users
    df_user_checkin['business_id']
    rec_list=rec_list[rec_list.apply(lambda row: False if row['business_id'] in list_user_pois else True,axis=1)]
    return rec_list
