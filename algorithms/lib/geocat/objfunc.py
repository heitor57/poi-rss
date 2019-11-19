# Objective functions of the DivGeoCat
import networkx as nx
import lib.heuristics as heuristics

def category_dis_sim(category1,category2,undirected_category_tree):
	dissim=0.0
	spd=nx.shortest_path_length(undirected_category_tree,category1,category2)
	sim = 1.0 / (1.0 + spd)
	dissim=1.0-sim
	return dissim

def min_dist_to_list_cat(poi_id,pois,poi_cats,undirected_category_tree):
	num_pois=len(pois)
	min_dissim=1.0
	cats_of_poi=poi_cats[poi_id]
	if num_pois==0:
		min_dissim=1.0
	else:
		for index_1 in pois:

			local_min_distance=1
			cur_distance=0
			for category1 in poi_cats[index_1]:
				for category2 in cats_of_poi:
					cur_distance=category_dis_sim(
						category1,
						category2,undirected_category_tree)
					local_min_distance=min(local_min_distance,cur_distance)
			min_dissim=min(min_dissim,local_min_distance)
	return min_dissim

def ld_objective_function(score,poi_id,pois,poi_cats,undirected_category_tree,div_weight):
	div=min_dist_to_list_cat(poi_id,pois,poi_cats,undirected_category_tree)
	return (1-div_weight)*score + div*div_weight

def ld(uid,training_matrix,tmp_rec_list,tmp_score_list,
			poi_cats,undirected_category_tree,K,div_weight):
	
	range_K=range(K)
	rec_list=[]
	
	final_scores=[]
	for i in range_K:
		#print(i)
		poi_to_insert=None
		max_objective_value=-200
		for j in range(len(tmp_rec_list)):
			candidate_poi_id=tmp_rec_list[j]
			candidate_score=tmp_score_list[j]
			objective_value=ld_objective_function(candidate_score,candidate_poi_id,rec_list,poi_cats,
													undirected_category_tree,div_weight)
			if objective_value > max_objective_value:
				max_objective_value=objective_value
				poi_to_insert=candidate_poi_id
		if poi_to_insert is not None:
			rm_idx=tmp_rec_list.index(poi_to_insert)
			tmp_rec_list.pop(rm_idx)
			tmp_score_list.pop(rm_idx)
			rec_list.append(poi_to_insert)
			final_scores.append(max_objective_value)
	
	return rec_list,final_scores
	
def gc(poi_id,rec_list,relevant_cats,poi_cats):

	cats=set(poi_cats[poi_id])
	for lid in rec_list:
		cats.update(poi_cats[lid])
	count_equal=0
	for cat1 in relevant_cats:
		for cat2 in cats:
			if cat1 == cat2:
				#print(cat1)
				count_equal=count_equal+1	
	return count_equal/len(relevant_cats)

def gc_list(rec_list,relevant_cats,poi_cats):
	gc(rec_list[-1],rec_list[:-1],relevant_cats,poi_cats)


def update_geo_cov(poi_id,log_poi_ids,rec_list_size,poi_cover,poi_neighbors,neighbors):
	log_size=len(log_poi_ids)

#	 neighbors=list()
#	 for id_neighbor in poi_neighbors[poi_id]:
#		 for i in range(log_size):
#			 log_poi_id=log_poi_ids[i]
#			 if log_poi_id == id_neighbor:
#				 neighbors.append(i)
	
	num_neighbors=len(neighbors)
	#set_trace()
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

def ILD_GC_PR(score,ild_div,gc_div,pr,current_proportionality,rec_list_size,div_geo_cat_weight,div_weight):

	delta_proportionality=max(0,pr-current_proportionality)
	
	#delta_proportionality=max(0,update_geo_cov(poi,df_user_review,rec_list_size,business_cover.copy(),poi_neighbors)-current_proportionality)
	#print(poi.business_id,ild_div,gc_div,delta_proportionality)

	if delta_proportionality<0:
		delta_proportionality=0
	div_cat = gc_div+ild_div/rec_list_size
	div_geo = delta_proportionality
	div=div_geo_cat_weight*div_geo+(1-div_geo_cat_weight)*div_cat
	return (score**(1-div_weight))*(div**div_weight)

def geocat_objective_function(poi_id,score,
								rec_list,rec_list_size,
								poi_cats,undirected_category_tree,relevant_cats,
								log_poi_ids,poi_cover,poi_neighbors,log_neighbors,
								div_geo_cat_weight,div_weight,current_proportionality):
	ild_div=min_dist_to_list_cat(poi_id,rec_list,poi_cats,undirected_category_tree)
	gc_div=gc(poi_id,rec_list,relevant_cats,poi_cats)
	pr=update_geo_cov(poi_id,log_poi_ids,rec_list_size,poi_cover.copy(),poi_neighbors,log_neighbors[poi_id])
	objective_value=ILD_GC_PR(score,ild_div,gc_div,pr,current_proportionality,rec_list_size,div_geo_cat_weight,div_weight)
	return objective_value


# new NORM ILD GC PR
def NORM_ILD_GC_PR(score,ild_div,gc_div,pr,current_proportionality,rec_list_size,div_geo_cat_weight,div_weight):

	delta_proportionality=max(0,pr-current_proportionality)
	
	#delta_proportionality=max(0,update_geo_cov(poi,df_user_review,rec_list_size,business_cover.copy(),poi_neighbors)-current_proportionality)
	#print(poi.business_id,ild_div,gc_div,delta_proportionality)

	if delta_proportionality<0:
		delta_proportionality=0
	div_cat = gc_div+ild_div/rec_list_size
	div_geo = delta_proportionality
	div=div_geo_cat_weight*div_geo+(1-div_geo_cat_weight)*div_cat
	return (score**(1-div_weight))*(div**div_weight)


def persongeocat_objective_function(poi_id,score,
								rec_list,rec_list_size,
								poi_cats,undirected_category_tree,relevant_cats,
								log_poi_ids,poi_cover,poi_neighbors,log_neighbors,
								div_geo_cat_weight,div_weight,current_proportionality):
	ild_div=min_dist_to_list_cat(poi_id,rec_list,poi_cats,undirected_category_tree)
	gc_div=gc(poi_id,rec_list,relevant_cats,poi_cats)
	pr=update_geo_cov(poi_id,log_poi_ids,rec_list_size,poi_cover.copy(),poi_neighbors,log_neighbors[poi_id])
	objective_value=NORM_ILD_GC_PR(score,ild_div,gc_div,pr,current_proportionality,rec_list_size,div_geo_cat_weight,div_weight)
	return objective_value

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
def geocat(uid,training_matrix,tmp_rec_list,tmp_score_list,
		  poi_cats,poi_neighbors,K,undirected_category_tree,
		  div_geo_cat_weight,div_weight,method='local_max'):
	range_K=range(K)
	rec_list=[]

	lids=training_matrix[uid].nonzero()[0]

	lid_visits=training_matrix[:,lids].sum(axis=0)
	mean_visits=lid_visits.mean()
	relevant_lids=lids[lid_visits>mean_visits]
	relevant_cats=set()
	for lid in relevant_lids:
		relevant_cats.update(poi_cats[lid])
	# log_size=training_matrix[0,training_matrix[0,:].nonzero()[0]].sum()
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
#		 print(uid)
#		 print("Count:",cnt)
	# div_geo_cat_weight = 0.75 # beta,this is here because of the work to be done on parameter customization for each user
	# div_weight = 0.5 # lambda, geo vs cat
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

	rec_list = final_scores = None

	if method == 'local_max':
		rec_list,final_scores = heuristics.local_max(tmp_rec_list, tmp_score_list, poi_cats, poi_neighbors,
				K, undirected_category_tree, rec_list, relevant_cats, log_poi_ids, log_neighbors, poi_cover,
				current_proportionality, div_geo_cat_weight, div_weight, final_scores)

	elif method == 'tabu_search':
		rec_list,final_scores = heuristics.tabu_search(tmp_rec_list, tmp_score_list, poi_cats,
				poi_neighbors, K, undirected_category_tree, relevant_cats, div_geo_cat_weight,
				div_weight, user_log)

	elif method == 'particle_swarm':
		rec_list,final_scores = heuristics.particle_swarm(tmp_rec_list, tmp_score_list, poi_cats,
				poi_neighbors, K, undirected_category_tree, relevant_cats, div_geo_cat_weight,
				div_weight, user_log)

	else:
		print('Warning! Invalid method choice:', method)
	
	return rec_list,final_scores
