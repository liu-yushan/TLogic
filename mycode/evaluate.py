import json
import argparse
import numpy as np

import rule_application as ra
from grapher import Grapher
from temporal_walk import store_edges
from baseline import baseline_candidates, calculate_obj_distribution


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", default="", type=str)
parser.add_argument("--test_data", default="test", type=str)
parser.add_argument("--candidates", "-c", default="", type=str)
parsed = vars(parser.parse_args())


def filter_candidates(test_query, candidates, test_data):
    """
    Filter out those candidates that are also answers to the test query
    but not the correct answer.

    Parameters:
        test_query (np.ndarray): test_query
        candidates (dict): answer candidates with corresponding confidence scores
        test_data (np.ndarray): test dataset

    Returns:
        candidates (dict): filtered candidates
    """

    other_answers = test_data[
        (test_data[:, 0] == test_query[0])
        * (test_data[:, 1] == test_query[1])
        * (test_data[:, 2] != test_query[2])
        * (test_data[:, 3] == test_query[3])
    ]

    if len(other_answers):
        objects = other_answers[:, 2]
        for obj in objects:
            candidates.pop(obj, None)

    return candidates


def calculate_rank(test_query_answer, candidates, num_entities, setting="best"):
    """
    Calculate the rank of the correct answer for a test query.
    Depending on the setting, the average/best/worst rank is taken if there
    are several candidates with the same confidence score.

    Parameters:
        test_query_answer (int): test query answer
        candidates (dict): answer candidates with corresponding confidence scores
        num_entities (int): number of entities in the dataset
        setting (str): "average", "best", or "worst"

    Returns:
        rank (int): rank of the correct answer
    """

    rank = num_entities
    if test_query_answer in candidates:
        conf = candidates[test_query_answer]
        all_confs = list(candidates.values())
        ranks = [idx for idx, x in enumerate(all_confs) if x == conf]
        if setting == "average":
            rank = (ranks[0] + ranks[-1]) // 2 + 1
        elif setting == "best":
            rank = ranks[0] + 1
        elif setting == "worst":
            rank = ranks[-1] + 1

    return rank


dataset = parsed["dataset"]
candidates_file = parsed["candidates"]
dir_path = "../output/" + dataset + "/"
dataset_dir = "../data/" + dataset + "/"
data = Grapher(dataset_dir)
num_entities = len(data.id2entity)
test_data = data.test_idx if (parsed["test_data"] == "test") else data.valid_idx
learn_edges = store_edges(data.train_idx)
obj_dist, rel_obj_dist = calculate_obj_distribution(data.train_idx, learn_edges)


all_candidates = json.load(open(dir_path + candidates_file))
all_candidates = {int(k): v for k, v in all_candidates.items()}
for k in all_candidates:
    all_candidates[k] = {int(cand): v for cand, v in all_candidates[k].items()}

hits_1 = 0
hits_3 = 0
hits_10 = 0
mrr = 0

num_samples = len(test_data)
print("Evaluating " + candidates_file + ":")
for i in range(num_samples):
    test_query = test_data[i]
    if all_candidates[i]:
        candidates = all_candidates[i]
    else:
        candidates = baseline_candidates(
            test_query[1], learn_edges, obj_dist, rel_obj_dist
        )
    candidates = filter_candidates(test_query, candidates, test_data)
    rank = calculate_rank(test_query[2], candidates, num_entities)

    if rank:
        if rank <= 10:
            hits_10 += 1
            if rank <= 3:
                hits_3 += 1
                if rank == 1:
                    hits_1 += 1
        mrr += 1 / rank

hits_1 /= num_samples
hits_3 /= num_samples
hits_10 /= num_samples
mrr /= num_samples

print("Hits@1: ", round(hits_1, 6))
print("Hits@3: ", round(hits_3, 6))
print("Hits@10: ", round(hits_10, 6))
print("MRR: ", round(mrr, 6))

filename = candidates_file[:-5] + "_eval.txt"
with open(dir_path + filename, "w", encoding="utf-8") as fout:
    fout.write("Hits@1: " + str(round(hits_1, 6)) + "\n")
    fout.write("Hits@3: " + str(round(hits_3, 6)) + "\n")
    fout.write("Hits@10: " + str(round(hits_10, 6)) + "\n")
    fout.write("MRR: " + str(round(mrr, 6)))
