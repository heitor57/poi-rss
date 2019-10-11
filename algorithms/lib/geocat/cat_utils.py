import pandas as pd
import networkx as nx
def get_most_detailed_categories(categories,dict_alias_title,dict_alias_depth):
    max_height=0
    for category in categories:
        max_height = max(dict_alias_depth[category],max_height)
    new_categories=list()
    for category in categories:
        height=dict_alias_depth[category]
        if(height == max_height):
            new_categories.append(category)
    return new_categories


def cat_structs():
    df_categories=pd.read_json("../data/categories.json")

    # dicionário alias title 2 way

    dict_alias_title=dict()
    for index, row in df_categories.iterrows():
        dict_alias_title[row['alias']]=row['title']
        dict_alias_title[row['title']]=row['alias']


    category_tree= nx.DiGraph()
    for index, row in df_categories.iterrows():
        if not row['parents']:
            category_tree.add_edge(row['alias'],'root') # root node if no parents
        else:
            for parent_label in row['parents']:
                category_tree.add_edge(row['alias'],parent_label)

    undirected_category_tree=category_tree.to_undirected()
    # dict alias depth
    dict_alias_depth=dict()
    for index, row in df_categories.iterrows():
        dict_alias_depth[row['alias']]=nx.shortest_path_length(category_tree,row['alias'],'root')

#     not_bounded_depth_items = set()
#     for category,depth in dict_alias_depth.items():
#        # print(category)
#         if not(depth <= 2):
#             not_bounded_depth_items.add(category)

#     for category in not_bounded_depth_items:
#         del dict_alias_depth[category]
#         category_tree.remove_node(category)
#         other_name = dict_alias_title[category]
#         del dict_alias_title[category]
#         del dict_alias_title[other_name]

    df_categories=None
    #len(dict_alias_depth),len(dict_alias_title),len(category_tree)

    return dict_alias_title,category_tree,dict_alias_depth