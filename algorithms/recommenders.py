import numpy as np
import pandas as pd

## Mostpopular rank by the top-k most visited pois by distinct users
def mostpopular(df_review,k):
    return df_review[['business_id','user_id']].groupby('business_id').count().sort_values('user_id',ascending=0).reset_index(level=0).rename(columns={"user_id":"score"}).head(k)
