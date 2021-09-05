import objfunc
from RecList import RecList
from Swarm import Swarm
from Particle import Particle
from random import randint
from sys import exit
from time import time

def local_max(tmp_rec_list, tmp_score_list, poi_cats, poi_neighbors, K, undirected_category_tree,
                rec_list, relevant_cats, log_poi_ids, log_neighbors, poi_cover, current_proportionality,
                div_geo_cat_weight, div_weight, final_scores, objective_function,div_cat_weight):

    range_K = range(K)
    current_gc = 0
    need_gc_diff = objective_function == objfunc.gc_diff_geocat_objective_function or\
                   objective_function == objfunc.gc_diff_og_geocat_objective_function
    need_div_cat_weight = objective_function == objfunc.cat_weight_geocat_objective_function
    for i in range_K:
        #print(i)
        poi_to_insert=None
        max_objective_value=-200
        for j in range(len(tmp_rec_list)):
            candidate_poi_id=tmp_rec_list[j]
            candidate_score=tmp_score_list[j]

            if need_div_cat_weight:
                objective_value = objective_function(candidate_poi_id,candidate_score,
                                                    rec_list,K,
                                                    poi_cats,undirected_category_tree,relevant_cats,
                                                    log_poi_ids,poi_cover,poi_neighbors,log_neighbors,
                                                    div_geo_cat_weight,div_weight,current_proportionality,
                                                     div_cat_weight)
            elif need_gc_diff:
                objective_value = objective_function(candidate_poi_id,candidate_score,
                                                    rec_list,K,
                                                    poi_cats,undirected_category_tree,relevant_cats,
                                                    log_poi_ids,poi_cover,poi_neighbors,log_neighbors,
                                                    div_geo_cat_weight,div_weight,current_proportionality,
                                                    current_gc)
            else:
                objective_value = objective_function(candidate_poi_id,candidate_score,
                                                    rec_list,K,
                                                    poi_cats,undirected_category_tree,relevant_cats,
                                                    log_poi_ids,poi_cover,poi_neighbors,log_neighbors,
                                                    div_geo_cat_weight,div_weight,current_proportionality)

            if objective_value > max_objective_value:
                max_objective_value=objective_value
                poi_to_insert=candidate_poi_id
            pass
        if poi_to_insert is not None:

            rm_idx=tmp_rec_list.index(poi_to_insert)

            tmp_rec_list.pop(rm_idx)
            tmp_score_list.pop(rm_idx)
            rec_list.append(poi_to_insert)
            final_scores.append(max_objective_value)
            # remove from tmp_rec_list
            current_proportionality=objfunc.update_geo_cov(poi_to_insert,log_poi_ids,K,poi_cover,poi_neighbors,log_neighbors[poi_to_insert])
            if need_gc_diff:
                current_gc = objfunc.gc_list(rec_list,relevant_cats,poi_cats)
            #print(current_proportionality)
    
    return rec_list,final_scores

def tabu_search(tmp_rec_list, tmp_score_list, poi_cats, poi_neighbors, K, undirected_category_tree,
                relevant_cats, div_geo_cat_weight, div_weight, user_log, div_cat_weight):

    max_iteration = 30
    iteration = 0
    neighbour_number = 10
    list_size = K
    tabu_size = 1000
    tabu_index = 0
    max_time = 30 # seconds
    # div_geo_cat_weight = 0.75 # beta,this is here because of the work to be done on parameter customization for each user
    # div_weight = 0.5 # lambda, geo vs cat

    current_solution = RecList(list_size)
    best_solution = RecList(list_size)

    # Inicializa a solução inicial
    current_solution.create_from_base_rec(tmp_rec_list, tmp_score_list)
    # Calcula a função objetivo
    current_solution.fo = metrics.calculate_fo(current_solution, poi_cats, undirected_category_tree,
                                               user_log, poi_neighbors, div_geo_cat_weight, div_weight, K, relevant_cats, div_cat_weight)
    
    # Inicializa a melhor solução
    best_solution.clone(current_solution)

    # Cria a lista tabu
    tabu_list = []
    # Adiciona a solução inicial à lista tabu
    tabu_list.append(current_solution)
    # start_time = time()
    while iteration < max_iteration:
        # Gera o primeiro vizinho
        new_solution = current_solution.create_neighbour(tmp_rec_list, len(tmp_rec_list), tmp_score_list)
       
        new_solution.fo = metrics.calculate_fo(new_solution, poi_cats, undirected_category_tree,
                                               user_log, poi_neighbors, div_geo_cat_weight, div_weight, K, relevant_cats, div_cat_weight)
        # Gera os outros n-1 vizinhos
        for i in range(neighbour_number-1):
            neighbour_solution = current_solution.create_neighbour(tmp_rec_list, len(tmp_rec_list), tmp_score_list)

            neighbour_solution.fo = metrics.calculate_fo(neighbour_solution, poi_cats, undirected_category_tree,
                                                         user_log, poi_neighbors, div_geo_cat_weight, div_weight, K, relevant_cats, div_cat_weight)
            if  neighbour_solution not in tabu_list and (neighbour_solution.fo > new_solution.fo or new_solution in tabu_list):
                # Atualiza o melhor vizinho
                new_solution.clone(neighbour_solution)
                
        # Calcula-se a função objetiva de ambas as soluções e mantêm-se a melhor:
        if new_solution not in tabu_list and new_solution.fo > current_solution.fo:

            current_solution.clone(new_solution) # Substitui
            tabu_list.append(current_solution) # Adiciona à lista tabu

            if len(tabu_list) > tabu_size:
                tabu_list.pop(0)
            
            if current_solution.fo > best_solution.fo:
                # Atualiza a melhor
                best_solution.clone(current_solution)
    
        iteration += 1

    # A melhor solução está em best_solution, agora não sei o que fazer para completar o processo de diversificação
    return best_solution.get_result()

def pso_roulette_shuffle(roulette_list, roulette_size):
    for i in range(roulette_size):
        r = randint(0, roulette_size-1)
        temp = roulette_list[i]
        roulette_list[i] = roulette_list[r]
        roulette_list[r] = temp

def pso_roulette(w, c1, c2):
    roulette_list = []
    t1 = int(w * 10)
    t2 = int(c1 * 10)
    t3 = 10 - (t1 + t2)

    for i in range(t1):
        roulette_list.append(1)
    
    for i in range(t2):
        roulette_list.append(2)

    for i in range(t3):
        roulette_list.append(3)

    pso_roulette_shuffle(roulette_list, len(roulette_list))
    roulette_position = randint(0, 9)
    return roulette_list[roulette_position]

def particle_swarm(tmp_rec_list, tmp_score_list, poi_cats, poi_neighbors, K, undirected_category_tree,
                   relevant_cats, div_geo_cat_weight, div_weight, user_log, div_cat_weight):
    
    swarm_size = 30
    particle_size = K
    base_rec_size = len(tmp_rec_list)
    iteration = 0
    max_iteration = 30
    # max_time = 30 # seconds
    # Global best solution
    global_best = RecList(particle_size)
    
    # Best diversity
    dbest = RecList(particle_size)

    # Particle swarm
    swarm = Swarm(swarm_size)
    swarm.create_particles(tmp_rec_list, tmp_score_list, particle_size, base_rec_size)
    cont = 0

    # Calculate local best for each particle and global best
    for i in range(swarm_size):
        metrics.pso_calculate_fo(swarm[i], poi_cats, undirected_category_tree, user_log, poi_neighbors,
                                 div_geo_cat_weight, div_weight, K, relevant_cats, dbest, div_cat_weight)

        # Update global best
        if (global_best.fo < swarm[i].best_fo):
            global_best.clone_particle(swarm[i])
    # start_time = time()
    while iteration < max_iteration:
        gbest_position = -1

        for i in range(swarm_size):
            # Build particle from parents
            new_particle = Particle(particle_size)

            for i in range(particle_size):
                item_id = -1
                item_score = -1

                while item_id == -1 or item_id in new_particle:
                    particle_choice = pso_roulette(0.3, 0.3, 0.4)
                    position = randint(0, particle_size-1)
                    
                    if (particle_choice == 1):
                        item_id = swarm[i].item_list[position]
                        item_score = swarm[i].score_list[position]
                    elif (particle_choice == 2):
                        item_id = swarm[i].best_item_list[position]
                        item_score = swarm[i].best_score_list[position]
                    else:
                        item_id = global_best.item_list[position]
                        item_score = global_best.score_list[position]

                new_particle.add_item(item_id, item_score)
            
            swarm[i].clone_new_current(new_particle)

            # Calcule local best and global best
            metrics.pso_calculate_fo(swarm[i], poi_cats, undirected_category_tree, user_log, poi_neighbors,
                                     div_geo_cat_weight, div_weight, K, relevant_cats, dbest, div_cat_weight)
            # Update global best
            if (global_best.fo < swarm[i].best_fo):
                global_best.clone_particle(swarm[i])

        # Path relink function is not complete
        # global_best = path_Relink(gbest, gbestPos, dBest, swarm, hashFeature, numPreds, alfa, featureSize);
        iteration += 1

    return global_best.get_result()
