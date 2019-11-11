import geocat.objfunc as objfunc
from RecList import RecList
import metrics

def local_max(tmp_rec_list, tmp_score_list, poi_cats, poi_neighbors, K, undirected_category_tree,
				rec_list, relevant_cats, log_poi_ids, log_neighbors, poi_cover, current_proportionality,
				div_geo_cat_weight, div_weight, final_scores):

	range_K = range(K)

	for i in range_K:
			#print(i)
			poi_to_insert=None
			max_objective_value=-200
			for j in range(len(tmp_rec_list)):
				candidate_poi_id=tmp_rec_list[j]
				candidate_score=tmp_score_list[j]
				ild_div=objfunc.min_dist_to_list_cat(candidate_poi_id,rec_list,poi_cats,undirected_category_tree)
				gc_div=objfunc.gc(candidate_poi_id,rec_list,relevant_cats,poi_cats)
				pr=objfunc.update_geo_cov(candidate_poi_id,log_poi_ids,K,poi_cover.copy(),poi_neighbors,log_neighbors[candidate_poi_id])

				objective_value=objfunc.ILD_GC_PR(candidate_score,ild_div,gc_div,pr,current_proportionality,K,div_geo_cat_weight,div_weight)
				#print(candidate_poi_id,ild_div,gc_div,max(0,pr-current_proportionality),objective_value)
				#print(candidate_poi_id,objective_value)

				if objective_value > max_objective_value:
					max_objective_value=objective_value
					poi_to_insert=candidate_poi_id
				pass
			if poi_to_insert is not None:
				#print(poi_to_insert,max_objective_value)

				rm_idx=tmp_rec_list.index(poi_to_insert)

				tmp_rec_list.pop(rm_idx)
				tmp_score_list.pop(rm_idx)
				rec_list.append(poi_to_insert)
				final_scores.append(max_objective_value)
				# remove from tmp_rec_list
				current_proportionality=objfunc.update_geo_cov(poi_to_insert,log_poi_ids,K,poi_cover,poi_neighbors,log_neighbors[poi_to_insert])
				#print(current_proportionality)
	
	return rec_list,final_scores

def tabu_search(tmp_rec_list, tmp_score_list, poi_cats, poi_neighbors, K, undirected_category_tree,
				rec_list, relevant_cats, log_poi_ids, log_neighbors, poi_cover, current_proportionality,
				div_geo_cat_weight, div_weight, final_scores, user_id, training_matrix, user_log):

	max_iteration = 100
	iteration = 0
	neighbour_number = 20
	list_size = K
	tabu_size = 100
	tabu_index = 0

	# div_geo_cat_weight = 0.75 # beta,this is here because of the work to be done on parameter customization for each user
	# div_weight = 0.5 # lambda, geo vs cat

	current_solution = RecList(list_size)
	best_solution = RecList(list_size)

	# Inicializa a solução inicial
	current_solution.create_from_base_rec(tmp_rec_list, tmp_score_list)
	# Calcula a função objetivo
	current_solution.fo = metrics.calculate_fo(current_solution, poi_cats, undirected_category_tree,
						user_log, poi_neighbors, div_geo_cat_weight, div_weight, K, relevant_cats)
	
	# Inicializa a melhor solução
	best_solution.clone(current_solution)

	# Cria a lista tabu
	tabu_list = []
	# Adiciona a solução inicial à lista tabu
	tabu_list.append(current_solution)

	while iteration < max_iteration:
		# Gera o primeiro vizinho
		new_solution = current_solution.create_neighbour(tmp_rec_list, len(tmp_rec_list), tmp_score_list)
		
		# Gera os outros n-1 vizinhos
		for i in range(neighbour_number-1):
			neighbour_solution = current_solution.create_neighbour(tmp_rec_list, len(tmp_rec_list), tmp_score_list)

			if  neighbour_solution not in tabu_list and (neighbour_solution.fo > new_solution.fo or new_solution in tabu_list):
				# Atualiza o melhor vizinho
				new_solution.clone(neighbour_solution)
				
		# Calcula-se a função objetiva de ambas as soluções e mantêm-se a melhor:
		if new_solution not in tabu_list and new_solution.fo > current_solution.fo:

			current_solution.clone(new_solution) # Substitui
			tabu_list[tabu_index] = current_solution # Adiciona à lista tabu

			if tabu_index == tabu_size-1:
				tabu_index = 0
			else:
				tabu_index += 1
			
			if current_solution.fo > best_solution.fo:
				# Atualiza a melhor
				best_solution.clone(current_solution)
	
		iteration += 1

	# A melhor solução está em best_solution, agora não sei o que fazer para completar o processo de diversificação
	return best_solution.get_result()