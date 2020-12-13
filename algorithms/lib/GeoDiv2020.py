import numpy as np
import igraph

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
                    distance = int(_dist(coo1, coo2)*10)
                    distribution[distance] += 1
        total = 1.0 * sum(distribution.values())
        for distance in distribution:
            distribution[distance] /= total
        distribution = sorted(distribution.items(), key=lambda k: k[0])
        return zip(*distribution)

    def pr_d(self, d):
        d = max(0.01, d)
        return self.a * (d ** self.b)

    def train(self,training_matrix, poi_coos):

        x, t = self.compute_distance_distribution(check_in_matrix, poi_coos)
        x_0 = x[0]
        t_0 = t[0]
        x = np.log2(x[1:])
        t = np.log2(t[1:])
        w0, w1 = np.random.random(), np.random.random()
        max_iterations = 12000
        lambda_w = 0.1
        alpha = 1e-5
        for iteration in range(max_iterations):
            ew = 0.0
            d_w0, d_w1 = 0.0, 0.0
            d_w0 = np.sum(w0 + w1 * x - t)
            d_w1 = np.sum((w0 + w1 * x - t) * x)
            w0 -= alpha * (d_w0 + lambda_w * w0)
            w1 -= alpha * (d_w1 + lambda_w * w1)
            # ew += 0.5 * (w0 + w1 * x[n] - t[n])**2
            # ew += 0.5 * lambda_w * (w0**2 + w1**2)

        self.a, self.b = 2**w0, w1
        th_far = 0
        acc_value = 0
        for i in range(600):
            if acc_value >= self.threshold_area:
                break
            acc_value+=self.pr_d(th_far*10)
            th_far += 0.1
        self.th_far = th_far

        print(f"a {self.a} b {self.b} x_0 {x_0} th_far {th_far}")
        
    def active_area_selection(self,uid,training_matrix, poi_coos):
        user_log = training_matrix[uid]
        num_total_checkins = np.count_nonzero(user_log)
        pois_ids = np.nonzero(user_log)[0]
        # log_poi_ids=list()
        # for lid in user_log.nonzero()[0]:
        #     for visits in range(int(user_log[lid])):
        #         log_poi_ids.append(lid)
        # vertices = np.arange(len(log_poi_ids))
        g = igraph.Graph()
        g.add_vertices(map(str,list(range(len(pois_ids)))))
        for i in range(len(pois_ids)):
            for j in range(len(pois_ids)):
                if j > i and _dist(poi_coos[pois_ids[i]],poi_coos[pois_ids[j]]) <= 2*self.th_far:
                    g.add_edges([(str(i),str(j))])

        graphs = []
        for i in range(len(pois_ids)):
            vertices = [j.name for j in g.bfsiter(i)]
            new_g = igraph.Graph()
            new_g.add_vertices(vertices)
            for edge in [(vertices[j],vertices[j+1]) for j in range(len(vertices)-1)]:
                new_g.add_edge(edge)

            # to_delete_ids = [v.index for v in G.vs if '@enron.com' not in v['label']]
            g.delete_vertices(vertices)
            if len(vertices) > 1:
                graphs.append(new_g)

        graphs_order = [len(i.vs) for i in graphs]
        sorted_lists = reversed(sorted(zip(graphs_order,graphs)))
        graphs =  [element for _, element in sorted_lists]
        areas = set()
        num_checkins = 0
        for new_g in graphs:
            areas.add(list(map(lambda x: int(x),new_g.vs['name'])))
            
            num_checkins += len(new_g.vs)
            if num_checkins/num_total_checkins >= self.threshold_area:
                break

        return areas
        

    def predict(self):
        pass
    
def update_geo_cov(poi_id,log_poi_ids,rec_list_size,poi_cover,poi_neighbors,neighbors):
    log_size=len(log_poi_ids)
    num_neighbors=len(neighbors)
    vl=1
    COVER_OF_POI=log_size/rec_list_size
    accumulated_cover=0
    # Cover calc
    if num_neighbors<1:
        accumulated_cover+=COVER_OF_POI
    else:
        cover_of_neighbor=COVER_OF_POI/num_neighbors
        for index in neighbors:
            poi_cover[index]+=cover_of_neighbor
    accumulated_cover/=log_size
    # end PR and DP

    DP=0
    
    for i in range(log_size):
        lid=i
        if vl>=poi_cover[lid]:
            DP+=(vl-poi_cover[lid])**2
    DP+=(accumulated_cover**2)/2
    DP_IDEAL=log_size+0.5
    PR=1-DP/(DP_IDEAL)
    
    return PR
def geodiv_objective_function(poi_id,log_poi_ids,rec_list_size,poi_cover,poi_neighbors,neighbors,current_proportionality,div_weight,score):
    pr=update_geo_cov(poi_id,log_poi_ids,rec_list_size,poi_cover,poi_neighbors,neighbors)
    div=max(0,pr-current_proportionality)
    return (score**(1-div_weight))*(div**div_weight)

def geodiv(uid,training_matrix,tmp_rec_list,tmp_score_list,
            poi_neighbors,K,div_weight):
    
    range_K=range(K)
    rec_list=[]
    
    user_log=training_matrix[uid]
    #print(mean_visits)
    log_poi_ids=list()
    poi_cover=list()
    for lid in user_log.nonzero()[0]:
        for visits in range(int(user_log[lid])):
            poi_cover.append(0)
            log_poi_ids.append(lid)
    log_size=len(log_poi_ids)
    assert user_log[user_log.nonzero()[0]].sum() == len(poi_cover)

    current_proportionality=0
    final_scores=[]
    log_neighbors=dict()
    for poi_id in tmp_rec_list:
        neighbors=list()
        for id_neighbor in poi_neighbors[poi_id]:
            for i in range(log_size):
                log_poi_id=log_poi_ids[i]
                if log_poi_id == id_neighbor:
                    neighbors.append(i)
        log_neighbors[poi_id]=neighbors
    for i in range_K:
        #print(i)
        poi_to_insert=None
        max_objective_value=-200
        for j in range(len(tmp_rec_list)):
            candidate_poi_id=tmp_rec_list[j]
            candidate_score=tmp_score_list[j]
            #objective_value=geodiv_objective_function(candidate_poi_id,log_poi_id,K,poi_cover.copy(),poi_neighbors,log_neighbors[candidate_poi_id],current_proportionality,div_weight,candidate_score)
            objective_value=geodiv_objective_function(candidate_poi_id,log_poi_ids,K,poi_cover.copy(),poi_neighbors,log_neighbors[candidate_poi_id],
                                        current_proportionality,div_weight,candidate_score)
            
            if objective_value > max_objective_value:
                max_objective_value=objective_value
                poi_to_insert=candidate_poi_id
        if poi_to_insert is not None:
            rm_idx=tmp_rec_list.index(poi_to_insert)
            tmp_rec_list.pop(rm_idx)
            tmp_score_list.pop(rm_idx)
            rec_list.append(poi_to_insert)
            final_scores.append(max_objective_value)
            current_proportionality=update_geo_cov(poi_to_insert,log_poi_ids,K,poi_cover,poi_neighbors,log_neighbors[poi_to_insert])
    
    return rec_list,final_scores
