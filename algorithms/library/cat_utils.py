import pandas as pd
import networkx as nx
import collections
import numpy as np
from igraph import *


def get_most_detailed_categories(categories, dict_alias_title,
                                 dict_alias_depth):
    max_height = 0
    for category in categories:
        max_height = max(dict_alias_depth[category], max_height)
    new_categories = list()
    for category in categories:
        height = dict_alias_depth[category]
        if (height == max_height):
            new_categories.append(category)
    return new_categories


def cat_structs_igraph(catfilename):
    df_categories = pd.read_json(catfilename)

    dict_alias_title = dict()
    for index, row in df_categories.iterrows():
        dict_alias_title[row['alias']] = row['title']
        dict_alias_title[row['title']] = row['alias']

    category_tree = Graph(directed=False)
    category_tree.add_vertices(list(df_categories['alias']))
    category_tree.add_vertex('root')

    for index, row in df_categories.iterrows():
        if not row['parents']:
            category_tree.add_edge(row['alias'],
                                   'root')  # root node if no parents
        else:
            for parent_label in row['parents']:
                category_tree.add_edge(row['alias'], parent_label)

    dict_alias_depth = dict()
    for index, row in df_categories.iterrows():
        dict_alias_depth[row['alias']] = category_tree.shortest_paths(
            row['alias'], 'root')[0][0]

    df_categories = None

    return dict_alias_title, category_tree, dict_alias_depth


def cat_structs(catfilename):
    df_categories = pd.read_json(catfilename)  #"../data/categories.json"

    # dicionário alias title 2 way

    dict_alias_title = dict()
    for index, row in df_categories.iterrows():
        dict_alias_title[row['alias']] = row['title']
        dict_alias_title[row['title']] = row['alias']

    category_tree = nx.DiGraph()
    for index, row in df_categories.iterrows():
        if not row['parents']:
            category_tree.add_edge(row['alias'],
                                   'root')  # root node if no parents
        else:
            for parent_label in row['parents']:
                category_tree.add_edge(row['alias'], parent_label)

    undirected_category_tree = category_tree.to_undirected()
    # dict alias depth
    dict_alias_depth = dict()
    for index, row in df_categories.iterrows():
        dict_alias_depth[row['alias']] = nx.shortest_path_length(
            category_tree, row['alias'], 'root')

    df_categories = None

    return dict_alias_title, category_tree, dict_alias_depth


def get_users_cat_visits(training_matrix, poi_cats):
    users_cv = []
    for i in range(training_matrix.shape[0]):
        cats_visits = collections.defaultdict(int)
        lids = training_matrix[i].nonzero()[0]
        for lid in lids:
            for cat in poi_cats[lid]:
                cats_visits[cat] += training_matrix[i, lid]
        cats_visits = dict(cats_visits)
        users_cv.append(cats_visits)
    return users_cv
