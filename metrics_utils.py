import torch
import numpy as np
from sklearn.metrics import ndcg_score
from typing import List


def precision_at_k(expected_order, actual_order, k):
    """
    Precision at k calculation. Expected order is an array
    of places (integer numbers): lower values mean more important items.
    Examples:
        [1, 2, 3, 4] -> [4, 3, 2, 1] for k=2 results in 0
        [1, 2, 3, 4] -> [1, 2, 3, 4] for k=2 results in 1

    :param expected_order: real ranking from ground-truth
    :param actual_order: order to test
    :param k: number of places to take into account
    :return: float value of precision between 0 and 1
    """
    if expected_order.shape != actual_order.shape:
        raise Exception("Shapes must match")
    if len(expected_order.shape) != 1:
        raise Exception("Single dimension array expected. Consider reshaping")

    p = 0
    for i in range(1, k + 1):
        idx = np.where(actual_order == i)
        if expected_order[idx] <= k:
            p += 1 / k
    return p


def map_at_k(expected_order, actual_order, k):
    """
    Mean average precision calculation. Expected order is an array
    of places (integer numbers): lower values mean more important items.
    Examples:
        [1, 2, 3, 4] -> [4, 3, 2, 1] for k=2 results in 0
        [1, 2, 3, 4] -> [1, 2, 3, 4] for k=2 results in 1

    :param expected_order: real ranking from ground-truth
    :param actual_order: order to test
    :param k: number of places to take into account
    :return: float value of precision between 0 and 1
    """
    if expected_order.shape != actual_order.shape:
        raise Exception("Shapes must match")
    if len(expected_order.shape) != 1:
        raise Exception("Single dimension array expected. Consider reshaping")

    p = 0
    t = 0
    for i in range(1, k + 1):
        idx = np.where(actual_order == i)
        if expected_order[idx] <= k:
            t += 1
            p += t / i
    return p / k


def ndcg_at_k(expected_order, actual_scores, k):
    """
    NDCG score provided by the sklearn ndcg_score method:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html

    The expected order is the array of places (integer number >0): lower values
    mean more important items. The actual score is the score computed for each
    element. Higher score means more important item.

    :param expected_order: real ranking from ground-truth
    :param actual_scores: scores to test
    :param k: number of places in the rank to take into account
    :return: float value of precision between 0 and 1
    """
    if expected_order.shape != actual_scores.shape:
        raise Exception("Shapes must match")
    if len(expected_order.shape) != 1:
        raise Exception("Not tested on higher dimensions")

    expected_order = np.expand_dims(expected_order, axis=0)
    actual_scores = np.expand_dims(actual_scores, axis=0)

    ndcg_true_rel = (np.max(expected_order) - expected_order)
    ndcg_real_scores = actual_scores
    return ndcg_score(ndcg_true_rel, ndcg_real_scores, k=k)


def torch_scores_to_order(scores):
    """
    Converts torch tensor of scores to torch tensor of places. Higher score
    leads to lower place (1st place is the best).

    :param scores: 1-d tensor of assigned scores
    :return: tensor of the same shape with places
    """
    if len(scores.shape) != 1:
        raise Exception("Expecting a 1-d tensor")

    s = torch.argsort(scores, descending=True)
    r = torch.zeros(scores.shape, dtype=torch.long)
    for i in range(scores.shape[-1]):
        r[s[i]] = i
    return r + 1


class Metrics:
    def __init__(self, k_max=10):
        self.k_max = k_max
        self.loss = 0
        self.p_at_k = np.zeros(k_max)
        self.map_at_k = np.zeros(k_max)
        self.ndcg_at_k = np.zeros(k_max)
        self.number = 0

    def add_item(self, loss, expected_order, actual_scores):
        """
        Add metrics item. Expected order and actual scores are torch tensors

        :param loss:
        :param expected_order:
        :param actual_scores: 1-d tensor of scores
        """
        if expected_order.shape != actual_scores.shape:
            raise Exception("Expected and actual shapes must match")
        if len(actual_scores.shape) != 1:
            raise Exception("Expected 1-d array for the scores")
        if actual_scores.shape[0] < self.k_max:
            raise Exception("The array is less than k max value. Cannot compute metrics")

        actual_order = torch_scores_to_order(actual_scores)
        for k in range(self.k_max):
            self.p_at_k[k] += precision_at_k(expected_order, actual_order, k + 1)
            self.map_at_k[k] += map_at_k(expected_order, actual_order, k + 1)
            self.ndcg_at_k[k] += ndcg_at_k(expected_order, actual_scores, k + 1)
        self.loss += loss
        self.number += 1

    def get_mean_loss(self):
        return self.loss / self.number

    def get_p_k(self):
        return self.p_at_k / self.number

    def get_map_k(self):
        return self.map_at_k / self.number

    def get_ndcg_k(self):
        return self.ndcg_at_k / self.number


def merge_metrics(list_of_metrics: List[Metrics]) -> Metrics:
    k_max = list_of_metrics[0].k_max
    result = Metrics(k_max=k_max)

    for m in list_of_metrics:
        if m.k_max != k_max:
            raise Exception("Metrics dimensions must match {0} != {1}".format(m.k_max, k_max))

        for k in range(k_max):
            result.p_at_k[k] += m.p_at_k[k]
            result.map_at_k[k] += m.map_at_k[k]
            result.ndcg_at_k[k] += m.ndcg_at_k[k]

        result.loss += m.loss
        result.number += m.number

    return result
