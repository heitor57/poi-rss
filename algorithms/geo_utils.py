import numpy as np
import math
def haversine(lat1, lon1, lat2, lon2):
    # deg to rad
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2-lon1
    dlat = lat2-lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2*np.arcsin(np.sqrt(a))
    r = 6371 # km
    return c*r

VERY_SMALL_DEGREE = 0.00001
def mercator(lat1,lon1,lat2,lon2):
#     using lat2 and lon2 as numpy array to gain performance
    #if abs(lat1 - lat2) < VERY_SMALL_DEGREE and abs(lon1 - lon2) < VERY_SMALL_DEGREE:
      #  return 0.0

    theta = lon1 - lon2
    dist = np.sin(np.radians(lat1)) * np.sin(np.radians(lat2)) + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(np.radians(theta))
    
    dist = np.arccos(dist)
    
    dist = np.degrees(dist)
    dist = dist * 60 * 1.1515
    
    dist = dist * 1.609344


    return dist