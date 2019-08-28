### ADAPTATIVE KERNEL ESTIMATOR
## not used
def band_width_h1(df_user_poi_count,N,n):
    sum_latitude_pois_weighted=df_user_poi_count[['count','latitude']].\
    apply(lambda x: x['count']*x['latitude'],axis=1).sum()
    
    sum_latitude_pois_weighted_normalized=sum_latitude_pois_weighted/N
    
    sum_latitude_pois_weighted_squared=df_user_poi_count[['count','latitude']].apply(
        lambda x: (x['count']*x['latitude']-sum_latitude_pois_weighted_normalized)**2,axis=1).sum()
    
    return 1.06*(n**(-1/5))*math.sqrt(sum_latitude_pois_weighted_squared/N)

def band_width_h2(df_user_poi_count,N,n):
    sum_latitude_pois_weighted=df_user_poi_count[['count','longitude']].\
    apply(lambda x: x['count']*x['longitude'],axis=1).sum()
    
    sum_latitude_pois_weighted_normalized=sum_latitude_pois_weighted/N
    
    sum_latitude_pois_weighted_squared=df_user_poi_count[['count','longitude']].apply(
        lambda x: (x['count']*x['longitude']-sum_latitude_pois_weighted_normalized)**2,axis=1).sum()
    
    return 1.06*(n**(-1/5))*math.sqrt(sum_latitude_pois_weighted_squared/N)

def normal_kernel_kh(poi,poii,h1,h2):
    return (1/(2*math.pi*h1*h2))*math.exp(
        -(poi['latitude']-poii['latitude'])**2/(2*(h1**2))-(poi['longitude']-poii['longitude'])**2/(2*(h2**2))
    )
def kernel_density_estimation_geo(poi,df_user_poi_count,N,n,h1,h2):
    return df_user_poi_count[['count','latitude','longitude']].apply(
        lambda poii: poii['count']*normal_kernel_kh(poi,poii,h1,h2),axis=1).sum()/N

def local_bandwith_hi(poi,df_user_poi_count,N,n,h1,h2,sensivity_parameter):
    ''' sensivity_parameter \in [0,1]'''
    geometric_mean=df_user_poi_count[['count','latitude','longitude']].apply(
    lambda x: kernel_density_estimation_geo(x,df_user_poi_count,N,n,h1,h2),axis=1).prod()**(1/n)
    return ((geometric_mean**(-1))*kernel_density_estimation_geo(poi,df_user_poi_count,N,n,h1,h2))**(-sensivity_parameter)

def adaptative_kernel_kh_hi(poi,poii,df_user_poi_count,N,n,h1,h2,sensivity_parameter):
    local_bandwith=local_bandwith_hi(poii,df_user_poi_count,N,n,h1,h2,sensivity_parameter)
    return (1/(2*math.pi*h1*h2*(local_bandwith**2)))*math.exp(
        -(poi['latitude']-poii['latitude'])**2/(2*((h1**2)*(local_bandwith**2)))\
        -(poi['longitude']-poii['longitude'])**2/(2*((h2**2)*(local_bandwith**2))))

def variables_adaptative_kernel_estimation_geo(df_user_review):
    # N: total of checkins of user, n: total unique checkins of user
    N=df_user_review.business_id.count()
    n=df_user_review.business_id.nunique()
    # dataframe with count column, count: number of visits
    df_user_poi_count=df_user_review.copy()
    df_user_poi_count.drop_duplicates(subset=['business_id'],inplace=True)
    df_user_poi_count['count']=df_user_review.groupby('business_id')['user_id'].transform('count')
    # band width calculation
    h1=band_width_h1(df_user_poi_count,N,n)
    h2=band_width_h2(df_user_poi_count,N,n)
    return [df_user_poi_count,N,n,h1,h2]

def adaptative_kernel_estimation_geo(poi,df_user_poi_count,N,n,h1,h2,sensivity_parameter):
    
    return df_user_poi_count[['count','latitude','longitude']].apply(
        lambda poii: poii['count']*adaptative_kernel_kh_hi(poi,poii,df_user_poi_count,N,n,h1,h2,sensivity_parameter)
        ,axis=1).sum()/N

SENSIVITY_PARAMETER=0.5

### SUBAREA VISIT PROBABILITY

def subarea_visit_probability(subarea_pois,df_user_poi_count,N,n,h1,h2):
    return subarea_pois.apply(
        lambda poi:adaptative_kernel_estimation_geo(poi,df_user_poi_count,N,n,h1,h2,SENSIVITY_PARAMETER),axis=1).sum()

user_id=users[0]

df_user_review=df_review_training[df_review_training['user_id']==user_id]

vars_ake_geo=variables_adaptative_kernel_estimation_geo(df_user_review)
vars_ake_geo[0]=vars_ake_geo[0].loc[0:6]
# Visit probability to subareas
subareas_visit_probability=vars_ake_geo[0].groupby('subarea_id').apply(
    subarea_visit_probability,*vars_ake_geo).to_frame().rename(columns={0:"probability"})

### ACTIVITY DENSITY TO AREA


def user_activity_density_to_area(subarea_id,subareas_visit_probability):
    subarea_visit_probability=subareas_visit_probability.loc[subarea_id]['probability']
    min_visit_probability=subareas_visit_probability['probability'].max()
    max_visit_probability=subareas_visit_probability['probability'].min()
    return math.log(subarea_visit_probability-min_visit_probability+1)/math.log(
    max_visit_probability-min_visit_probability+1)

subareas_visit_probability['activity_density']=subareas_visit_probability.apply(
    lambda subarea: user_activity_density_to_area(subarea.name,subareas_visit_probability),axis=1)
subareas_visit_probability['activity_district']=subareas_visit_probability.apply(
    lambda subarea: True if subarea['activity_density']>=0.9 else False,axis=1)
subareas_visit_probability

