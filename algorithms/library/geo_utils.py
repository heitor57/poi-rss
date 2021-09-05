import numpy as np
import math
from numba import jit

@jit(nopython=True)
def haversine(lat1, lon1, lat2, lon2):
    # deg to rad
    # lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    lon1, lat1, lon2, lat2 = np.radians(lon1), np.radians(lat1), np.radians(lon2), np.radians(lat2)
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

def dist(loc1, loc2):
    lat1, long1 = loc1[0], loc1[1]
    lat2, long2 = loc2[0], loc2[1]
    if abs(lat1 - lat2) < 1e-6 and abs(long1 - long2) < 1e-6:
        return 0.0
    degrees_to_radians = math.pi/180.0
    phi1 = (90.0 - lat1)*degrees_to_radians
    phi2 = (90.0 - lat2)*degrees_to_radians
    theta1 = long1*degrees_to_radians
    theta2 = long2*degrees_to_radians
    cos = (math.sin(phi1)*math.sin(phi2)*math.cos(theta1 - theta2) +
           math.cos(phi1)*math.cos(phi2))
    arc = math.acos( cos )
    earth_radius = 6371
    return arc * earth_radius

def km_to_lat(km):
    return km/dist((0,0),(0,1))
