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
        # print(x)
        # print(t)
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
        self.th_far = np.round(th_far,1)
        self.th_far_radius = self.th_far/constants.EARTH_RADIUS

        print(f"a {self.a} b {self.b} t_0 {t_0} th_far {th_far}")
        dists = np.round(np.append(np.arange(0,self.th_far,0.1),self.th_far),1)
        self.accumulated_dist_probabilities = {dists[i]: v for i, v in enumerate(np.cumsum(list(map(self.pr_d,dists))))}
        print(self.accumulated_dist_probabilities)
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

        # print(graphs)
        graphs_order = [len(i.vs) for i in graphs]
        # sorted_lists = 
        # try:
        graphs =  [element for _, element in reversed(sorted(zip(graphs_order,graphs), key=lambda pair: pair[0]))]
        # except:
        #     print("GR ORDER:",graphs_order)
        #     print("GRs :",graphs)
        #     print("ZIPP:",list(zip(graphs_order,graphs)))
        #     graphs =  [element for _, element in reversed(sorted(list(zip(graphs_order,graphs))))]
        areas_lids = []
        num_checkins = 0
        for new_g in graphs:
            # print(set(map(lambda x: int(x),new_g.vs['name'])))
            tmp = list(map(lambda x: int(x),new_g.vs['name']))
            areas_lids.extend(tmp)
            # print(areas_lids)
            
            num_checkins += len(new_g.vs)
            if num_checkins/num_total_checkins >= self.threshold_area:
                break
            
        return set(areas_lids)

    def objective(self,uid,rec_list,final_scores,req_u,user_valid_lids,K,ideal_dp_u,closeness_user_log_lids_to_candidates_lids,summed_closeness_candidates_lids,old_pr):
        pr=self.pr(uid,rec_list,req_u,user_valid_lids,K,ideal_dp_u,closeness_user_log_lids_to_candidates_lids,summed_closeness_candidates_lids)
        div=max(0,pr-old_pr)
        return (final_scores[-1])**(1-self.div_weight)*(div**self.div_weight), pr

    def pr(self,uid,rec_list,req_u,user_valid_lids,K,ideal_dp_u,closeness_user_log_lids_to_candidates_lids,summed_closeness_candidates_lids):
        sum_quantities_checkins=0
        not_nearby = 0
        for lid in rec_list:
            not_nearby += summed_closeness_candidates_lids[lid] == 0
        # not_nearby = np.count_nonzero(list(summed_closeness_candidates_lids.values()))
        for log_lid in user_valid_lids:
            p_c_r = self.quantity(uid,rec_list, log_lid,closeness_user_log_lids_to_candidates_lids,summed_closeness_candidates_lids)
            if req_u >= p_c_r:
                sum_quantities_checkins += ((req_u-p_c_r)**2)*self.training_matrix[uid,log_lid]
            
                
        dp_u = sum_quantities_checkins + 0.5*(not_nearby/K)**2
        return 1 - dp_u/ideal_dp_u



    def quantity(self,uid,rec_list, log_lid,closeness_user_log_lids_to_candidates_lids,summed_closeness_candidates_lids):
        proximities = [self.proximity_to_checkins(uid,lid, log_lid,closeness_user_log_lids_to_candidates_lids,summed_closeness_candidates_lids)
         for lid in rec_list]
        return np.sum(proximities)
    # p'(c,l)
    def proximity_to_checkins(self,uid,lid, log_lid,closeness_user_log_lids_to_candidates_lids,summed_closeness_candidates_lids):
        # self.closeness(_dist(self.poi_coos[lid],self.poi_coos[]))
        user_log_lids = closeness_user_log_lids_to_candidates_lids.keys()
        # print(lid,summed_closeness_candidates_lids[lid])
        if closeness_user_log_lids_to_candidates_lids[log_lid][lid] == 0:
            return 0

        return closeness_user_log_lids_to_candidates_lids[log_lid][lid]/\
            summed_closeness_candidates_lids[lid]
        

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
        user_valid_lids = np.array(list(areas_lids))
        user_all_consumed_lids = set(np.nonzero(self.training_matrix[uid])[0])
        lids_original_indexes = {lid:i for i,lid in enumerate(tmp_rec_list)}

        # try:
        if len(areas_lids) == 0:
            pois_in_areas = set()
            print(f"User {uid} doenst have any area so it is returning top-{K} of base recommender")
            return tmp_rec_list[:K], tmp_score_list[:K]
        else:
            pois_in_areas = self.poi_coos_balltree.query_radius(np.array([self.poi_coos[lid] for lid in areas_lids])*np.pi/180,2*self.th_far_radius)
        # except:
        #     for lid in areas_lids:
        #         print(uid,self.poi_coos[lid])
        #     print(user_all_consumed_lids,areas_lids)
        #     raise SystemExit
        tmp = set()
        for i in pois_in_areas:
           tmp = tmp.union(i)
        pois_in_areas = tmp - user_all_consumed_lids
        candidate_pois = pois_in_areas.intersection(tmp_rec_list)
        # print("cand:",candidate_pois)
        if len(candidate_pois) < K:
            print(f"Need to complete, added {K-len(candidate_pois)} user {uid}")
            available_lids = list(set(tmp_rec_list) - candidate_pois)
            candidate_pois = candidate_pois.union(available_lids[:K-len(candidate_pois)])

        candidates_lids = list(candidate_pois) # mutate while recommending
        candidates_scores = [tmp_score_list[lids_original_indexes[lid]] for lid in candidates_lids]


        closeness_user_log_lids_to_candidates_lids = {lid_1: {lid_2: self.closeness(_dist(self.poi_coos[lid_1],self.poi_coos[lid_2])) for lid_2 in candidates_lids} for lid_1 in user_valid_lids}

        summed_closeness_candidates_lids = {lid_1:
                                            np.sum([closeness_user_log_lids_to_candidates_lids[lid_2][lid_1] for lid_2 in closeness_user_log_lids_to_candidates_lids.keys()])
                                            for lid_1 in candidates_lids}

        # assert(not any(lid in user_all_consumed_lids for lid in tmp_rec_list))
        # assert(not any(lid in user_all_consumed_lids for lid in list(summed_closeness_candidates_lids.keys())))
            
        # if uid < 20:
        #     print({lid_1:
        #            all(np.array([closeness_user_log_lids_to_candidates_lids[lid_2][lid_1] for lid_2 in closeness_user_log_lids_to_candidates_lids.keys()]) == 0)
        #            for lid_1 in candidates_lids})
        num_valid_checkins = np.sum(self.training_matrix[uid,user_valid_lids])
        req_u = K/num_valid_checkins
        ideal_dp_u = num_valid_checkins*req_u**2+0.5
        final_scores=[]
        rec_list = []
        old_pr = 0

        # print(candidates_lids)
        # raise SystemExit
        for i in range_K:
            #print(i)
            poi_to_insert=None
            max_objective_value=-200
            for j in range(len(candidates_lids)):
                candidate_poi_id=candidates_lids[j]
                candidate_score=candidates_scores[j]
                objective_value, pr = self.objective(uid,rec_list+[candidate_poi_id],final_scores+[candidate_score],
                                                 req_u,user_valid_lids,K,ideal_dp_u,
                                                 closeness_user_log_lids_to_candidates_lids,
                                                 summed_closeness_candidates_lids,old_pr)

                if objective_value > max_objective_value:
                    max_objective_value=objective_value
                    poi_to_insert=candidate_poi_id

            if poi_to_insert is not None:
                rm_idx=candidates_lids.index(poi_to_insert)
                candidates_lids.pop(rm_idx)
                candidates_scores.pop(rm_idx)
                rec_list.append(poi_to_insert)
                final_scores.append(max_objective_value)
                objective_value, old_pr = self.objective(uid,rec_list,final_scores,req_u,user_valid_lids,K,ideal_dp_u,closeness_user_log_lids_to_candidates_lids,summed_closeness_candidates_lids,old_pr)
                # current_proportionality=update_geo_cov(poi_to_insert,log_poi_ids,K,poi_cover,poi_neighbors,log_neighbors[poi_to_insert])

        return rec_list,final_scores
