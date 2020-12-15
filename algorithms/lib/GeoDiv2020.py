import numpy as np
import igraph
import math
from collections import defaultdict
import sklearn.neighbors
import constants

def _dist(loc1, loc2):
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

class GeoDiv2020:
    def __init__(self,div_weight):
        self.div_weight = div_weight
        self.threshold_area = 0.8

    @staticmethod
    def compute_distance_distribution(check_in_matrix, poi_coos):
        distribution = defaultdict(int)
        for uid in range(check_in_matrix.shape[0]):
            lids = check_in_matrix[uid, :].nonzero()[0]
            for i in range(len(lids)):
                for j in range(i+1, len(lids)):
                    lid1, lid2 = lids[i], lids[j]
                    coo1, coo2 = poi_coos[lid1], poi_coos[lid2]
                    distance = np.round(_dist(coo1, coo2))
                    distribution[distance] += 1

        total = 1.0 * sum(distribution.values())
        for distance in distribution:
            distribution[distance] /= total

        distribution = sorted(distribution.items(), key=lambda k: k[0])
        return zip(*distribution)

    def pr_d(self, d):
        # d = max(0.01, d)
        if d < 0.1: # There is no way with d<0.1
            return self.zero_probability
        p= min(self.a * ((d*10) ** self.b),1)
        return p


    def train(self,training_matrix, poi_coos):

        x, t = self.compute_distance_distribution(training_matrix, poi_coos)
        print(x)
        print(t)
        x_0 = x[0]
        t_0 = t[0]
        x = np.log10(x[1:])
        t = np.log10(t[1:])
        # w0, w1 = np.random.random(), np.random.random()
        w0, w1 = 0.1, -0.1
        # w0 = x_0
        max_iterations = 12000
        lambda_w = 0.1
        alpha = 0.000001
        for iteration in range(max_iterations):
            # d_w0, d_w1 = 0.0, 0.0
            d_w0 = np.sum(w0 + w1 * x - t)
            d_w1 = np.sum((w0 + w1 * x - t) * x)
            w0 -= alpha * (d_w0 + lambda_w * w0)
            w1 -= alpha * (d_w1 + lambda_w * w1)
            # if iteration > max_iterations-20:
            #     ew = 0.5 * (w0 + w1 * x - t)**2
            #     ew = np.sum(ew)+ 0.5 * lambda_w * (w0**2 + w1**2)
            #     print(ew,np.sqrt(np.mean((w0 + w1 * x - t)**2)))

        self.a, self.b = 10**w0, w1
        # print("+-----")
        # print(list(map(self.pr_d,np.arange(0,10,0.1))))
        # print("=-----")
        self.zero_probability = t_0
        th_far = 0
        acc_value = 0
        for i in range(600):
            if acc_value >= self.threshold_area:
                break
            acc_value+=self.pr_d(th_far)
            th_far += 0.1
        self.th_far = th_far
        self.th_far_radius = self.th_far/constants.EARTH_RADIUS

        print(f"a {self.a} b {self.b} t_0 {t_0} th_far {th_far}")
        self.accumulated_dist_probabilities = np.cumsum(list(map(self.pr_d,np.append(np.arange(0,th_far,0.1),th_far))))
        # self.dist_probabilities = np.cumsum(np.arange(0,th_far))

        num_lids = len(list(poi_coos.keys()))
        poi_coos_ = np.zeros((num_lids,2))
        for lid in poi_coos.keys():
            poi_coos_[lid] = poi_coos[lid]
        self.poi_coos_balltree = sklearn.neighbors.BallTree(np.array(poi_coos_)*np.pi/180,metric="haversine")
        self.training_matrix = training_matrix
        self.poi_coos = poi_coos
        
    def active_area_selection(self,uid):
        training_matrix = self.training_matrix
        poi_coos = self.poi_coos

        user_log = training_matrix[uid]
        num_total_checkins = np.count_nonzero(user_log)
        pois_ids = np.nonzero(user_log)[0]

        g = igraph.Graph()
        g.add_vertices(map(str,pois_ids))
        # self.poi_coos_balltree.query_radius([poi_coos[lid] for lid in poi_ids],2*self.th_far)
        for i in range(len(pois_ids)):
            for j in range(len(pois_ids)):
                if j > i and _dist(poi_coos[pois_ids[i]],poi_coos[pois_ids[j]]) <= 2*self.th_far:
                    g.add_edges([(str(pois_ids[i]),str(pois_ids[j]))])

        graphs = []
        while len(g.vs) != 0:
            vertices = [j['name'] for j in g.bfsiter(0)] # first vertice
            new_g = igraph.Graph()
            new_g.add_vertices(vertices)
            for edge in [(str(vertices[j]),str(vertices[j+1])) for j in range(len(vertices)-1)]:
                new_g.add_edge(*edge)

            # to_delete_ids = [v.index for v in G.vs if '@enron.com' not in v['label']]
            g.delete_vertices(vertices)
            if len(vertices) > 1:
                graphs.append(new_g)

        graphs_order = [len(i.vs) for i in graphs]
        sorted_lists = reversed(sorted(zip(graphs_order,graphs)))
        graphs =  [element for _, element in sorted_lists]
        areas_lids = set()
        num_checkins = 0
        for new_g in graphs:
            areas_lids.union(list(map(lambda x: int(x),new_g.vs['name'])))
            
            num_checkins += len(new_g.vs)
            if num_checkins/num_total_checkins >= self.threshold_area:
                break
            
        return areas_lids

    def objective(self,uid,rec_list,req_u,user_valid_lids,K,ideal_dp_u,closeness_user_log_lids_to_candidates_lids,summed_closeness_candidates_lids,old_pr):
        pr=self.pr(uid,rec_list,req_u,user_valid_lids,K,ideal_dp_u,closeness_user_log_lids_to_candidates_lids,summed_closeness_candidates_lids)
        div=max(0,pr-old_pr)
        return (score**(1-div_weight))*(div**div_weight)

    def pr(self,uid,rec_list,req_u,user_valid_lids,K,ideal_dp_u,closeness_user_log_lids_to_candidates_lids,summed_closeness_candidates_lids):
        sum_quantities_checkins=0
        not_nearby = np.count_nonzero(list(summed_closeness_candidates_lids.values()))
        for log_lid in user_valid_lids:
            p_c_r = self.quantity(uid,rec_list, log_lid,closeness_user_log_lids_to_candidates_lids,summed_closeness_candidates_lids)
            if req_u >= p_c_r:
                sum_quantities_checkins += ((req_u-p_c_r)**2)*self.training_matrix[uid,log_lid]
            
                
        dp_u = sum_quantities_checkins + 0.5*(not_nearby/K)**2
        return 1 - dp_u/ideal_dp_u



    def quantity(self,uid,rec_list, log_lid,closeness_user_log_lids_to_candidates_lids,summed_closeness_candidates_lids):
        proximities = [proximity_to_checkins(self,uid,lid, log_lid,closeness_user_log_lids_to_candidates_lids,summed_closeness_candidates_lids)
         for lid in rec_list]
        return np.sum(proximities)
    # p'(c,l)
    def proximity_to_checkins(self,uid,lid, log_lid,closeness_user_log_lids_to_candidates_lids,summed_closeness_candidates_lids):
        closeness(_dist(self.poi_coos[lid],self.poi_coos))
        user_log_lids = closeness_user_log_lids_to_candidates_lids.keys()
        return closeness_user_log_lids_to_candidates_lids[log_lid][lid]/\
            [summed_closeness_candidates_lids[lid]]
        

    def closeness(self,dist_km):
        dist_km = np.round(dist_km,1)
        if dist_km > self.th_far:
            return 0
        else:
            return 1-self.accumulated_dist_probabilities[dist_km]/self.accumulated_dist_probabilities[self.th_far]

    def predict(self,uid,tmp_rec_list,tmp_score_list,K):

        range_K=range(K)
        rec_list=[]
        areas_lids = self.active_area_selection(uid)
        lids_original_indexes = {lid:i for i,lid in enumerate(tmp_rec_list)}

        pois_in_areas = self.poi_coos_balltree.query_radius([poi_coos[lid] for lid in areas_lids],2*self.th_far_radius)
        pois_in_areas = set(pois_in_areas)
        candidate_pois = pois_in_areas.intersection(tmp_rec_list)
        if len(candidate_pois) < K:
            available_lids = candidate_pois - set(tmp_rec_list)
            candidate_pois.union(available_lids[K-len(candidate_pois)])

        candidate_lids = candidate_pois # mutate while recommending
        candidate_scores = [tmp_score_list[lids_original_indexes[lid]] for lid in candidate_lids]


        user_valid_lids = pois_in_areas.copy() # doenst mutate
        closeness_user_log_lids_to_candidates_lids = {lid_1: {lid_2: self.closeness(_dist(lid_1,lid_2)) for lid_2 in candidate_lids} for lid_1 in areas_lids}

        summed_closeness_candidates_lids = {lid_1:
                                            np.sum([closeness_user_log_lids_to_candidates_lids[lid_2][lid_1] for lid_2 in closeness_user_log_lids_to_candidates_lids.keys()])
                                            for lid_1 in candidate_lids}

        num_valid_checkins = np.sum(self.training_matrix[uid,user_valid_lids])
        req_u = K/num_valid_checkins
        ideal_dp_u = num_valid_checkins*req_u**2+0.5
        # user_log=training_matrix[uid]
        #print(mean_visits)
        # log_poi_ids=list()
        # poi_cover=list()
        # for lid in user_log.nonzero()[0]:
        #     for visits in range(int(user_log[lid])):
        #         poi_cover.append(0)
        #         log_poi_ids.append(lid)
        # log_size=len(log_poi_ids)
        # assert user_log[user_log.nonzero()[0]].sum() == len(poi_cover)

        # current_proportionality=0
        # final_scores=[]
        # log_neighbors=dict()
        # for poi_id in tmp_rec_list:
        #     neighbors=list()
        #     for id_neighbor in poi_neighbors[poi_id]:
        #         for i in range(log_size):
        #             log_poi_id=log_poi_ids[i]
        #             if log_poi_id == id_neighbor:
        #                 neighbors.append(i)
        #     log_neighbors[poi_id]=neighbors
        old_pr = 0
        for i in range_K:
            #print(i)
            poi_to_insert=None
            max_objective_value=-200
            for j in range(len(candidate_lids)):
                candidate_poi_id=candidate_lids[j]
                candidate_score=candidate_scores[j]
                #objective_value=geodiv_objective_function(candidate_poi_id,log_poi_id,K,poi_cover.copy(),poi_neighbors,log_neighbors[candidate_poi_id],current_proportionality,div_weight,candidate_score)
                # objective_value=geodiv_objective_function(candidate_poi_id,log_poi_ids,K,poi_cover.copy(),poi_neighbors,log_neighbors[candidate_poi_id],
                #                             current_proportionality,div_weight,candidate_score)
                objective_value = objective(self,uid,rec_list,req_u,user_valid_lids,K,ideal_dp_u,closeness_user_log_lids_to_candidates_lids,summed_closeness_candidates_lids,old_pr)

                if objective_value > max_objective_value:
                    max_objective_value=objective_value
                    poi_to_insert=candidate_poi_id

            if poi_to_insert is not None:
                rm_idx=candidate_lids.index(poi_to_insert)
                candidate_lids.pop(rm_idx)
                candidate_scores.pop(rm_idx)
                rec_list.append(poi_to_insert)
                final_scores.append(max_objective_value)
                old_pr = objective(self,uid,rec_list,req_u,user_valid_lids,K,ideal_dp_u,closeness_user_log_lids_to_candidates_lids,summed_closeness_candidates_lids,old_pr)
                # current_proportionality=update_geo_cov(poi_to_insert,log_poi_ids,K,poi_cover,poi_neighbors,log_neighbors[poi_to_insert])

        return rec_list,final_scores
    
# def update_geo_cov(poi_id,log_poi_ids,rec_list_size,poi_cover,poi_neighbors,neighbors):
#     log_size=len(log_poi_ids)
#     num_neighbors=len(neighbors)
#     vl=1
#     COVER_OF_POI=log_size/rec_list_size
#     accumulated_cover=0
#     # Cover calc
#     if num_neighbors<1:
#         accumulated_cover+=COVER_OF_POI
#     else:
#         cover_of_neighbor=COVER_OF_POI/num_neighbors
#         for index in neighbors:
#             poi_cover[index]+=cover_of_neighbor
#     accumulated_cover/=log_size
#     # end PR and DP

#     DP=0
    
#     for i in range(log_size):
#         lid=i
#         if vl>=poi_cover[lid]:
#             DP+=(vl-poi_cover[lid])**2
#     DP+=(accumulated_cover**2)/2
#     DP_IDEAL=log_size+0.5
#     PR=1-DP/(DP_IDEAL)
    
#     return PR
# def geodiv_objective_function(poi_id,log_poi_ids,rec_list_size,poi_cover,poi_neighbors,neighbors,current_proportionality,div_weight,score):
#     pr=update_geo_cov(poi_id,log_poi_ids,rec_list_size,poi_cover,poi_neighbors,neighbors)
#     div=max(0,pr-current_proportionality)
#     return (score**(1-div_weight))*(div**div_weight)

