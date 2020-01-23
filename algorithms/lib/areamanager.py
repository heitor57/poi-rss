import math
import numpy as np
def delimiter_area(case):
    if case == "lasvegas":
        print('Area selected: Las Vegas')
        city = "LasVegas"
        area = {}
        area['city'] = city.lower()
        area['final_latitude'] = 36.389326
        area['initial_latitude'] = 36.123935
        area['initial_longitude'] = -115.427600
        area['final_longitude'] = -115.048827
    elif case == "phoenix": 
        print('Area selected: Phoenix')
        city = "Phoenix"
        area = {}
        area['city'] = city.lower()
        area['final_latitude'] = 34.003012
        area['initial_latitude'] = 33.006796
        area['initial_longitude'] = -112.606674
        area['final_longitude'] = -111.381699
    #34.995653, -81.034521 35.400766, -80.651372
    elif case == "charlotte": 
        print('Area selected: Charlotte')
        city = "Charlotte"
        area = {}
        area['city'] = city.lower()
        area['final_latitude'] = 35.400766
        area['initial_latitude'] = 34.995653
        area['initial_longitude'] = -81.034521
        area['final_longitude'] = -80.651372
    elif case == "madison":
        print('Area selected: Madison')
        city = "Madison"
        area = {}
        area['city'] = city.lower()
        area['final_latitude'] = 43.215156
        area['initial_latitude'] = 42.936791
        area['initial_longitude'] = -89.608990
        area['final_longitude'] = -89.179837
    elif case == "montreal":
        print('Area selected: Montreal')
        city = "Montreal"
        area = {}
        area['city'] = city.lower()
        area['initial_latitude'] = 45.349779
        area['initial_longitude'] = -74.024676
        area['final_latitude'] = 45.817165
        area['final_longitude'] = -73.339345
    elif case == "pittsburgh":
        print('Area selected: %s' % (case))
        city = case
        area = {}
        area['city'] = city.lower()
        area['initial_latitude'] = 40.359118
        area['initial_longitude'] = -80.102733
        area['final_latitude'] = 40.502851
        area['final_longitude'] = -79.854168

    return area
def poi_in_area(area,poi):
    if (poi['latitude']>=area['initial_latitude']) and\
                (poi['latitude']<=area['final_latitude'])and\
                (poi['longitude']>=area['initial_longitude']) and\
                (poi['longitude']<=area['final_longitude']):
        return True
    return False
def pois_in_area(area,df_business):
    return df_business[(df_business['latitude']>=area['initial_latitude']) &\
                (df_business['latitude']<=area['final_latitude'])&\
                (df_business['longitude']>=area['initial_longitude']) &\
                (df_business['longitude']<=area['final_longitude'])]
def poi_set_subarea(area,df_business_in_area,distance_km):
    '''
        area: is for example a city like phoenix, las vegas, new york, etc.
        df_business_in_area: is a dataframe of business filtered in a area
        distance_km: is the distance of subareas or area in km^2
    '''
    longitude_delta = abs(area['final_longitude']-area['initial_longitude'])
    latitude_delta = abs(area['final_latitude']-area['initial_latitude'])
    avg_latitude = (area['initial_latitude']+area['final_latitude'])/2.0
    LAT_DEGREE_KM_EQUATOR=111.0 # 110.57
    LONG_DEGREE_KM_EQUATOR=111.321 # 111.32
    # define step degree in latitude
    subarea_latitude_delta_degree = distance_km/LAT_DEGREE_KM_EQUATOR
    # define step degree in longitude
    subarea_longitude_delta_degree = distance_km/(LONG_DEGREE_KM_EQUATOR * math.cos(avg_latitude * math.pi/180))
    # number of subareas
    num_subareas = math.ceil(longitude_delta/subarea_longitude_delta_degree) * math.ceil(latitude_delta/subarea_latitude_delta_degree)
    
    df_business_in_area['subarea_id']=\
    np.abs(np.ceil((df_business_in_area['longitude']-area['final_longitude'])/subarea_longitude_delta_degree))+\
    (np.abs(np.ceil((df_business_in_area['latitude']-area['initial_latitude'])/subarea_latitude_delta_degree))-1)\
    * (np.ceil(longitude_delta/subarea_longitude_delta_degree))
#    Code that explains the above logic 
#
#     for index,row in df_business_in_area.iterrows():
#         latitude_poi_in_subarea = (row['latitude']-area['initial_latitude'])/subarea_latitude_delta_degree
#         longitude_poi_in_subarea = (row['longitude']-area['final_longitude'])/subarea_longitude_delta_degree
#         line = abs(math.ceil(latitude_poi_in_subarea))            
#         column = abs(math.ceil(longitude_poi_in_subarea))
#         subarea_id = column + (line -1) * (math.ceil(longitude_delta/subarea_longitude_delta_degree))
#         row['subarea_id']=subarea_id
    return df_business_in_area
