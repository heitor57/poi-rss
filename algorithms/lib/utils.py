import os
from enum import Enum
import numpy as np

import scipy
# def get_id_to_int(df_checkin):
#     poi_int_id=df_checkin.reset_index()['business_id'].drop_duplicates().reset_index(drop=True).to_dict()
#     poi_num=len(poi_int_id)
#     user_int_id=df_checkin.reset_index().user_id.drop_duplicates().reset_index(drop=True).to_dict()
#     user_num=len(user_int_id)
#     for i,j in poi_int_id.copy().items():
#         poi_int_id[j]=i
#     for i,j in user_int_id.copy().items():
#         user_int_id[j]=i
#     return user_int_id,poi_int_id
# def transform_id_to_int(df_checkin,user_int_id,poi_int_id):
#     df_checkin['user_id']=df_checkin['user_id'].apply(lambda user_id:user_int_id[user_id])
#     df_checkin['business_id']=df_checkin['business_id'].apply(lambda poi_id:poi_int_id[poi_id])

def create_path_to_file(filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)


class StatisticResult(Enum):
    GAIN = 1
    LOSS = 2
    TIE = 3
    # ERROR = 4
def statistic_test(x, y, p):
    # try:
    statistic, pvalue = scipy.stats.ttest_ind(x, y)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    if pvalue < p:
        if x_mean > y_mean:
            return StatisticResult.GAIN
        else:
            return StatisticResult.LOSS
    else:
        return StatisticResult.TIE
    # except e:
    #     print(e)
    #     return 'error'