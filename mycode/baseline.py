from collections import Counter


def baseline_candidates(test_query_rel, edges, obj_dist, rel_obj_dist):
    """
    Define the answer candidates based on the object distribution as a simple baseline.

    Parameters:
        test_query_rel (int): test query relation
        edges (dict): edges from the data on which the rules should be learned
        obj_dist (dict): overall object distribution
        rel_obj_dist (dict): object distribution for each relation

    Returns:
        candidates (dict): candidates along with their distribution values
    """

    if test_query_rel in edges:
        candidates = rel_obj_dist[test_query_rel]
    else:
        candidates = obj_dist

    return candidates


def calculate_obj_distribution(learn_data, edges):
    """
    Calculate the overall object distribution and the object distribution for each relation in the data.

    Parameters:
        learn_data (np.ndarray): data on which the rules should be learned
        edges (dict): edges from the data on which the rules should be learned

    Returns:
        obj_dist (dict): overall object distribution
        rel_obj_dist (dict): object distribution for each relation
    """

    objects = learn_data[:, 2]
    dist = Counter(objects)
    for obj in dist:
        dist[obj] /= len(learn_data)
    obj_dist = {k: round(v, 6) for k, v in dist.items()}
    obj_dist = dict(sorted(obj_dist.items(), key=lambda item: item[1], reverse=True))

    rel_obj_dist = dict()
    for rel in edges:
        objects = edges[rel][:, 2]
        dist = Counter(objects)
        for obj in dist:
            dist[obj] /= len(objects)
        rel_obj_dist[rel] = {k: round(v, 6) for k, v in dist.items()}
        rel_obj_dist[rel] = dict(
            sorted(rel_obj_dist[rel].items(), key=lambda item: item[1], reverse=True)
        )

    return obj_dist, rel_obj_dist
