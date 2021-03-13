import unittest
import torch
import math

from exploded_logit import ExplodedLogitTransformation


# noinspection PyTypeChecker
class ExplodedLogitTransformationTest(unittest.TestCase):

    def test_forward_simple(self):
        context = type('ExplodedLogitTestContext', (), {})()
        scores = torch.tensor([1, 2, 3, 4], dtype=torch.float64)
        order = torch.tensor([3, 2, 4, 1], dtype=torch.long)

        actual = ExplodedLogitTransformation.forward(context, scores, order)

        expected = torch.tensor([[1, 1, 1, -math.inf],
                                 [2, 2, -math.inf, -math.inf],
                                 [3, 3, 3, 3],
                                 [4, -math.inf, -math.inf, -math.inf]], dtype=torch.float64)

        self.assertTrue(torch.equal(actual, expected), "Wrong forward pass")

    def test_forward_batch(self):
        context = type('ExplodedLogitTestContext', (), {})()
        scores = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=torch.float64)
        order = torch.tensor([[3, 2, 4, 1], [1, 2, 3, 4], [4, 1, 2, 3]], dtype=torch.long)

        actual = ExplodedLogitTransformation.forward(context, scores, order)

        expected = torch.tensor([[[1, 1, 1, -math.inf],
                                  [2, 2, -math.inf, -math.inf],
                                  [3, 3, 3, 3],
                                  [4, -math.inf, -math.inf, -math.inf]],

                                 [[5, -math.inf, -math.inf, -math.inf],
                                  [6, 6, -math.inf, -math.inf],
                                  [7, 7, 7, -math.inf],
                                  [8, 8, 8, 8]],

                                 [[9, 9, 9, 9],
                                  [10, -math.inf, -math.inf, -math.inf],
                                  [11, 11, -math.inf, -math.inf],
                                  [12, 12, 12, -math.inf]]], dtype=torch.float64)

        self.assertTrue(torch.equal(actual, expected), "Wrong batching forward pass")

    def test_backward_simple(self):
        context = type('ExplodedLogitTestContext', (), {})()
        scores = torch.tensor([1, 2, 3, 4], dtype=torch.float64)
        order = torch.tensor([3, 2, 4, 1], dtype=torch.long)
        ExplodedLogitTransformation.forward(context, scores, order)

        gradient = torch.ones((4, 4), dtype=torch.float64)
        actual, _ = ExplodedLogitTransformation.backward(context, gradient)

        self.assertTrue(torch.equal(actual, order.double()), "Wrong backward pass")

    def test_backward_batch(self):
        context = type('ExplodedLogitTestContext', (), {})()
        scores = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=torch.float64)
        order = torch.tensor([[3, 2, 4, 1], [1, 2, 3, 4], [4, 1, 2, 3]], dtype=torch.long)
        ExplodedLogitTransformation.forward(context, scores, order)

        gradient = torch.ones((3, 4, 4), dtype=torch.float64)
        actual, _ = ExplodedLogitTransformation.backward(context, gradient)

        self.assertTrue(torch.equal(actual, order.double()), "Wrong backward pass")

    def test_backward_each_value(self):
        context = type('ExplodedLogitTestContext', (), {})()
        scores = torch.tensor([1, 2, 3, 4], dtype=torch.float64)
        order = torch.tensor([3, 2, 4, 1], dtype=torch.long)
        ExplodedLogitTransformation.forward(context, scores, order)

        for i in range(4):
            for j in range(4):
                gradient = torch.zeros((4, 4), dtype=torch.float64)
                gradient[i, j] = 1

                expected = torch.zeros(4, dtype=torch.float64)
                if j < order[i]:
                    expected[i] = 1

                actual, _ = ExplodedLogitTransformation.backward(context, gradient)
                self.assertTrue(torch.equal(actual, expected),
                                "Wrong backward pass: [{0}, {1}]\n{2}\n{3}".format(i, j, actual, expected))


if __name__ == '__main__':
    unittest.main()
