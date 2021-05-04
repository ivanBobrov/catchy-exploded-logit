import unittest
import torch
import math

from exploded_logit import ExplodedLogitTransformation
from exploded_logit import ExplodedLogitLoss


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


class ExplodedLogitLossTest(unittest.TestCase):

    def test_build_target_simple(self):
        loss = ExplodedLogitLoss()

        order = torch.tensor([3, 1, 2, 4], dtype=torch.long)
        expected = torch.tensor([[0., 0., 1., 0.],
                                 [1., 0., 0., 0.],
                                 [0., 1., 0., 0.],
                                 [0., 0., 0., 1.]], dtype=torch.float64)

        actual = loss.build_target(order)
        self.assertTrue(torch.equal(actual, expected),
                        "Building simple target matrix failed:\nActual:\n{0}\nExpected:\n{1}".format(actual, expected))

    def test_build_target_batch(self):
        loss = ExplodedLogitLoss()

        order = torch.tensor([[3, 1, 2, 4],
                              [2, 1, 3, 4],
                              [4, 2, 3, 1]], dtype=torch.long)

        expected = torch.tensor([[[0., 0., 1., 0.],
                                 [1., 0., 0., 0.],
                                 [0., 1., 0., 0.],
                                 [0., 0., 0., 1.]],

                                 [[0., 1., 0., 0.],
                                  [1., 0., 0., 0.],
                                  [0., 0., 1., 0.],
                                  [0., 0., 0., 1.]],

                                 [[0., 0., 0., 1.],
                                  [0., 1., 0., 0.],
                                  [0., 0., 1., 0.],
                                  [1., 0., 0., 0.]]], dtype=torch.float64)

        actual = loss.build_target(order)
        self.assertTrue(torch.equal(actual, expected),
                        "Building batch target matrix failed:\nActual:\n{0}\nExpected:\n{1}".format(actual, expected))

    def test_simple_forward_pass_bce(self):
        loss = ExplodedLogitLoss(loss_type='bce', reduction='sum')

        scores = torch.tensor([1.2, 4.8, 0.2, 5.6, 7.4, 0.], dtype=torch.float64)
        order = torch.tensor([6, 5, 3, 4, 2, 1], dtype=torch.long)

        loss_expected = torch.tensor(17.9922, dtype=torch.float64)
        loss_actual = loss.forward(scores, order)
        self.assertTrue(torch.isclose(loss_actual, loss_expected, atol=1e-4),
                        "Forward pass not valid: {0} != {1}".format(loss_actual, loss_expected))

    def test_simple_forward_pass_nll(self):
        loss = ExplodedLogitLoss(loss_type='nll', reduction='sum')

        scores = torch.tensor([1.2, 4.8, 0.2, 5.6, 7.4, 0.], dtype=torch.float64)
        order = torch.tensor([6, 5, 3, 4, 2, 1], dtype=torch.long)

        loss_expected = torch.tensor(14.0236, dtype=torch.float64)
        loss_actual = loss.forward(scores, order)
        self.assertTrue(torch.isclose(loss_actual, loss_expected, atol=1e-4),
                        "Forward pass not valid: {0} != {1}".format(loss_actual, loss_expected))

    def test_simple_forward_pass_nll_top(self):
        loss = ExplodedLogitLoss(loss_type='nll', reduction='sum', top_n=3)

        scores = torch.tensor([1.2, 4.8, 0.2, 5.6, 7.4, 0.], dtype=torch.float64)
        order = torch.tensor([6, 5, 3, 4, 2, 1], dtype=torch.long)

        loss_expected = torch.tensor(13.6171, dtype=torch.float64)
        loss_actual = loss.forward(scores, order)
        self.assertTrue(torch.isclose(loss_actual, loss_expected, atol=1e-4),
                        "Forward pass not valid: {0} != {1}".format(loss_actual, loss_expected))

    def test_batch_forward_pass_bce(self):
        loss = ExplodedLogitLoss(loss_type='bce', reduction='sum')

        scores = torch.tensor([[1.2, 4.8, 0.2, 5.6, 7.4, 0.],
                               [1.2, 4.8, 0.2, 5.6, 7.4, 0.]], dtype=torch.float64)
        order = torch.tensor([[6, 5, 3, 4, 2, 1],
                              [6, 5, 3, 4, 2, 1]], dtype=torch.long)

        loss_expected = torch.tensor(17.9922 * 2, dtype=torch.float64)
        loss_actual = loss.forward(scores, order)
        self.assertTrue(torch.isclose(loss_actual, loss_expected, atol=1e-4),
                        "Forward pass not valid: {0} != {1}".format(loss_actual, loss_expected))

    def test_batch_forward_pass_nll(self):
        loss = ExplodedLogitLoss(loss_type='nll', reduction='sum')

        scores = torch.tensor([[1.2, 4.8, 0.2, 5.6, 7.4, 0.],
                               [1.2, 4.8, 0.2, 5.6, 7.4, 0.]], dtype=torch.float64)
        order = torch.tensor([[6, 5, 3, 4, 2, 1],
                              [6, 5, 3, 4, 2, 1]], dtype=torch.long)

        loss_expected = torch.tensor(14.0236 * 2, dtype=torch.float64)
        loss_actual = loss.forward(scores, order)
        self.assertTrue(torch.isclose(loss_actual, loss_expected, atol=1e-4),
                        "Forward pass not valid: {0} != {1}".format(loss_actual, loss_expected))

    def test_batch_forward_pass_nll_top(self):
        loss = ExplodedLogitLoss(loss_type='nll', reduction='sum', top_n=3)

        scores = torch.tensor([[1.2, 4.8, 0.2, 5.6, 7.4, 0.],
                               [1.2, 4.8, 0.2, 5.6, 7.4, 0.]], dtype=torch.float64)
        order = torch.tensor([[6, 5, 3, 4, 2, 1],
                              [6, 5, 3, 4, 2, 1]], dtype=torch.long)

        loss_expected = torch.tensor(13.6171 * 2, dtype=torch.float64)
        loss_actual = loss.forward(scores, order)
        self.assertTrue(torch.isclose(loss_actual, loss_expected, atol=1e-4),
                        "Forward pass not valid: {0} != {1}".format(loss_actual, loss_expected))

    def test_simple_backward_pass(self):
        loss = ExplodedLogitLoss(loss_type='bce', reduction='sum')

        scores = torch.tensor([1.2, 4.8, 0.2, 5.6, 7.4, 0.], dtype=torch.float64, requires_grad=True)
        order = torch.tensor([6, 5, 3, 4, 2, 1], dtype=torch.long)

        loss = loss.forward(scores, order)
        loss.backward()

        grad_expected = torch.tensor([0.0604,  0.4864, -1.0052,  0.3989,  1.0611, -1.0016], dtype=torch.float64)
        grad_actual = scores.grad

        self.assertTrue(torch.allclose(grad_actual, grad_expected, atol=1e-4),
                        "Gradient is not valid:\n{0}\n{1}".format(grad_actual, grad_expected))

    def test_batch_backward_pass(self):
        loss = ExplodedLogitLoss(loss_type='bce', reduction='sum')

        scores = torch.tensor([[1.2, 4.8, 0.2, 5.6, 7.4, 0.],
                               [1.2, 4.8, 0.2, 5.6, 7.4, 0.]], dtype=torch.float64, requires_grad=True)
        order = torch.tensor([[6, 5, 3, 4, 2, 1],
                              [6, 5, 3, 4, 2, 1]], dtype=torch.long)

        loss = loss.forward(scores, order)
        loss.backward()

        grad_actual = scores.grad
        grad_expected = torch.tensor([[0.0604,  0.4864, -1.0052,  0.3989,  1.0611, -1.0016]],
                                     dtype=torch.float64).repeat(2, 1)

        self.assertTrue(torch.allclose(grad_actual, grad_expected, atol=1e-4),
                        "Gradient is not valid:\n{0}\n{1}".format(grad_actual, grad_expected))


if __name__ == '__main__':
    unittest.main()
