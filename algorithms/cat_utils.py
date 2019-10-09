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
