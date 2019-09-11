#def precision(df_user_review_test,rec_list,k):
#	

def category_dis_sim(category1,category2,undirected_category_tree):
    dissim=0.0
    spd=nx.shortest_path_length(undirected_category_tree,category1,category2)
    sim = 1.0 / (1.0 + spd)
    dissim=1.0-sim
    return dissim
    

def ild(rec_list,dict_alias_title,undirected_category_tree):
    
    min_dissim=1.0
    rec_list_size=len(rec_list)
    local_ild=0
    local_ild_km=0
    count=0
    
    if rec_list_size==0:
        min_dissim=1.0
    else:
        for index_1,row_1 in rec_list.iterrows():
            for index_2,row_2 in rec_list.iterrows():
                if index_1 != index_2:
                    local_min_distance=1
                    cur_distance=0
                    for category1 in row_1['categories']:
                        for category2 in row_2['categories']:
                            cur_distance=category_dis_sim(
                                dict_alias_title[category1],
                                dict_alias_title[category2],undirected_category_tree)
                            #print(category1,category2,cur_distance,local_min_distance)
                            local_min_distance=min(local_min_distance,cur_distance)
                    #min_dissim=min(min_dissim,local_min_distance)
                    local_ild+=local_min_distance
                    count+=1
    
    return local_ild/count


