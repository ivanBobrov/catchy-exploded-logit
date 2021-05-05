import unittest
import numpy as np
import torch

from metrics_utils import precision_at_k
from metrics_utils import map_at_k
from metrics_utils import ndcg_at_k
from metrics_utils import torch_scores_to_order
from metrics_utils import Metrics
from metrics_utils import merge_metrics


class PrecisionAtKTest(unittest.TestCase):
    def test_simple(self):
        expected_order = np.asarray([1, 2, 3, 4, 5, 6])

        self.assertAlmostEqual(precision_at_k(expected_order, np.asarray([1, 2, 3, 4, 5, 6]), 3), 1.000, places=2)
        self.assertAlmostEqual(precision_at_k(expected_order, np.asarray([6, 5, 4, 3, 2, 1]), 3), 0.000, places=2)
        self.assertAlmostEqual(precision_at_k(expected_order, np.asarray([1, 2, 4, 3, 5, 6]), 3), 0.666, places=2)
        self.assertAlmostEqual(precision_at_k(expected_order, np.asarray([1, 5, 4, 3, 2, 6]), 3), 0.333, places=2)
        self.assertAlmostEqual(precision_at_k(expected_order, np.asarray([6, 5, 1, 3, 2, 4]), 3), 0.333, places=2)
        self.assertAlmostEqual(precision_at_k(expected_order, np.asarray([6, 5, 2, 3, 4, 1]), 3), 0.333, places=2)
        self.assertAlmostEqual(precision_at_k(expected_order, np.asarray([6, 5, 3, 4, 2, 1]), 3), 0.333, places=2)


class MeanAveragePrecisionTest(unittest.TestCase):
    def test_simple(self):
        expected_order = np.asarray([1, 2, 3, 4, 5, 6])

        self.assertAlmostEqual(map_at_k(expected_order, np.asarray([1, 2, 3, 4, 5, 6]), 3), 1.000, places=2)
        self.assertAlmostEqual(map_at_k(expected_order, np.asarray([6, 5, 4, 3, 2, 1]), 3), 0.000, places=2)
        self.assertAlmostEqual(map_at_k(expected_order, np.asarray([1, 2, 4, 3, 5, 6]), 3), 0.666, places=2)
        self.assertAlmostEqual(map_at_k(expected_order, np.asarray([1, 5, 4, 3, 2, 6]), 3), 0.333, places=2)
        self.assertAlmostEqual(map_at_k(expected_order, np.asarray([6, 5, 1, 3, 2, 4]), 3), 0.333, places=2)
        self.assertAlmostEqual(map_at_k(expected_order, np.asarray([6, 5, 2, 3, 4, 1]), 3), 0.166, places=2)
        self.assertAlmostEqual(map_at_k(expected_order, np.asarray([6, 5, 3, 4, 2, 1]), 3), 0.111, places=2)


class NDCGTest(unittest.TestCase):
    def test_simple(self):
        expected_order = np.asarray([1, 2, 3, 4, 5, 6])

        self.assertAlmostEqual(ndcg_at_k(expected_order, np.asarray([1, 2, 3, 4, 5, 6]), 3), 0.180, places=2)
        self.assertAlmostEqual(ndcg_at_k(expected_order, np.asarray([6, 5, 4, 3, 2, 1]), 3), 1.000, places=2)
        self.assertAlmostEqual(ndcg_at_k(expected_order, np.asarray([1, 2, 4, 3, 5, 6]), 3), 0.236, places=2)
        self.assertAlmostEqual(ndcg_at_k(expected_order, np.asarray([1, 5, 4, 3, 2, 6]), 3), 0.445, places=2)
        self.assertAlmostEqual(ndcg_at_k(expected_order, np.asarray([6, 5, 1, 3, 2, 4]), 3), 0.833, places=2)
        self.assertAlmostEqual(ndcg_at_k(expected_order, np.asarray([6, 5, 2, 3, 4, 1]), 3), 0.889, places=2)
        self.assertAlmostEqual(ndcg_at_k(expected_order, np.asarray([6, 5, 3, 4, 2, 1]), 3), 0.944, places=2)


class ScoresOrderTest(unittest.TestCase):
    def test_ascending(self):
        tensor = torch.tensor([0.5, 0.4, 0.3, 0.2, 0.1])
        order = torch_scores_to_order(tensor)

        expected_order = torch.tensor([1, 2, 3, 4, 5])
        self.assertTrue(torch.equal(expected_order, order), "Ascending ordering failed")

    def test_descending(self):
        tensor = torch.tensor([0.1, 0.2, 0.25, 0.3, 0.35])
        order = torch_scores_to_order(tensor)

        expected_order = torch.tensor([5, 4, 3, 2, 1])
        self.assertTrue(torch.equal(expected_order, order), "Ascending ordering failed")

    def test_random_order(self):
        tensor = torch.tensor([1.4, 3.2, 0.14, 0.98, 3.13])
        order = torch_scores_to_order(tensor)

        expected_order = torch.tensor([3, 1, 5, 4, 2])
        self.assertTrue(torch.equal(expected_order, order), "Ascending ordering failed")


class MetricsTest(unittest.TestCase):
    def test_loss_average(self):
        metrics = Metrics(k_max=2)

        metrics.add_item(0, torch.tensor([.0, .2]), torch.tensor([.0, .2]))
        metrics.add_item(2, torch.tensor([.0, .2]), torch.tensor([.0, .2]))
        metrics.add_item(4, torch.tensor([.0, .2]), torch.tensor([.0, .2]))
        metrics.add_item(6, torch.tensor([.0, .2]), torch.tensor([.0, .2]))

        self.assertAlmostEqual(3.0, metrics.get_mean_loss(), msg="Mean loss calculation failed")

    def test_precision_average(self):
        metrics = Metrics(k_max=4)

        metrics.add_item(0, torch.tensor([1, 2, 3, 4]), torch.tensor([.8, .7, .4, .2]))
        metrics.add_item(2, torch.tensor([1, 2, 3, 4]), torch.tensor([.1, .2, .4, .3]))

        self.assertAlmostEqual(1.0, metrics.get_mean_loss(), msg="Mean loss calculation failed")
        self.assertAlmostEqual(0.833, metrics.get_p_k()[2], places=2, msg="Precision at 3 calculation failed")
        self.assertAlmostEqual(0.777, metrics.get_map_k()[2], places=2, msg="mAP at 3 calculation failed")
        self.assertAlmostEqual(0.710, metrics.get_ndcg_k()[2], places=2, msg="NDCG at 3 calculation failed")

    def test_merge_metrics(self):
        metrics1 = Metrics(k_max=3)
        metrics2 = Metrics(k_max=3)
        metrics3 = Metrics(k_max=3)

        metrics1.add_item(0, torch.tensor([1, 2, 3, 4]), torch.tensor([.8, .7, .4, .2]))
        metrics2.add_item(2, torch.tensor([1, 2, 3, 4]), torch.tensor([.1, .2, .4, .3]))
        metrics3.add_item(4, torch.tensor([1, 2, 3, 4]), torch.tensor([.1, .4, .3, .2]))

        metrics = merge_metrics([metrics1, metrics2, metrics3])

        self.assertAlmostEqual(2.0, metrics.get_mean_loss(), msg="Mean loss calculation failed")
        self.assertAlmostEqual(0.777, metrics.get_p_k()[2], places=2, msg="Precision at 3 calculation failed")
        self.assertAlmostEqual(0.740, metrics.get_map_k()[2], places=2, msg="mAP at 3 calculation failed")
        self.assertAlmostEqual(0.657, metrics.get_ndcg_k()[2], places=2, msg="NDCG at 3 calculation failed")


if __name__ == '__main__':
    unittest.main()
