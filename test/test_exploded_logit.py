import unittest
import torch
from torch.utils.data import DataLoader
from test.artificial_dataset_generator import ArtificialDataset

from linear_model import LinearModel
from exploded_logit import ExplodedLogitLoss


class ExplodedLogitIntegrationTest(unittest.TestCase):
    def test_single_column_input(self):
        torch.manual_seed(24637882)

        dataset_size = 8000
        test_dataset_size = 1000
        data_columns = 6
        competitors = 8

        dataset_generator = ArtificialDataset(dataset_size, competitors, data_columns, rand_eps=1e-3)
        loader_iterator = iter(DataLoader(dataset_generator))

        linear_model = LinearModel(data_columns, 1)  # number of columns to score
        optimizer = torch.optim.Adam(params=linear_model.parameters())
        loss = ExplodedLogitLoss(loss_type='nll', top_n=3)

        for step in range(dataset_size):
            data, order = next(loader_iterator)
            optimizer.zero_grad()

            score = linear_model(data).squeeze(-1)

            loss_value = loss(score, order)
            loss_value.backward()
            optimizer.step()
            if step % 1000 == 0:
                print("Loss value: {0}".format(loss_value.item()))

        with torch.no_grad():
            for _ in range(test_dataset_size):
                data, expected_order = next(loader_iterator)

                score = linear_model(data).squeeze(-1)
                actual_order = get_sort_order(score)

                self.assertTrue(torch.equal(actual_order, expected_order),
                                "Order not equal:\n{0}\n{1}".format(actual_order, expected_order))

        print("\n\nLinear transformation weights matrix\n--------------------")
        print(linear_model.linear.weight)
        print("Linear transformation bias:")
        print(linear_model.linear.bias)

        print("\n\nOriginal coefficients\n--------------------")
        print(dataset_generator.coeffs[0])
        print("Original bias")
        print(dataset_generator.biases[0])

    def test_without_bias(self):
        torch.manual_seed(24637882)

        dataset_size = 8000
        test_dataset_size = 1000
        data_columns = 3
        competitors = 8

        dataset_generator = ArtificialDataset(dataset_size, competitors, data_columns, rand_eps=1e-3, bias=False)
        loader_iterator = iter(DataLoader(dataset_generator))

        linear_model = LinearModel(data_columns, 1, bias=False)  # number of columns to score
        optimizer = torch.optim.Adam(params=linear_model.parameters())
        loss = ExplodedLogitLoss(loss_type='bce')

        for step in range(dataset_size):
            data, order = next(loader_iterator)
            optimizer.zero_grad()

            score = linear_model(data).squeeze(-1)

            loss_value = loss(score, order)
            loss_value.backward()
            optimizer.step()
            if step % 1000 == 0:
                print("Loss value: {0}".format(loss_value.item()))

        with torch.no_grad():
            for _ in range(test_dataset_size):
                data, expected_order = next(loader_iterator)

                score = linear_model(data).squeeze(-1)
                actual_order = get_sort_order(score)

                self.assertTrue(torch.equal(actual_order, expected_order),
                                "Order not equal:\n{0}\n{1}".format(actual_order, expected_order))

        print("\n\nLinear transformation weights matrix\n--------------------")
        print(linear_model.linear.weight)

        print("\n\nInverted original coefficients\n--------------------")
        print(1 / dataset_generator.coeffs[0])

    def test_without_bias_with_regularization(self):
        torch.manual_seed(24637882)

        dataset_size = 20000
        test_dataset_size = 1000
        data_columns = 1
        random_columns = 15
        competitors = 8
        regularization_lambda = 0.05

        dataset_generator = ArtificialDataset(dataset_size, competitors, data_columns, rand_eps=1e-3,
                                              number_of_random_columns=random_columns, bias=False)
        loader_iterator = iter(DataLoader(dataset_generator))

        linear_model = LinearModel(data_columns + random_columns, 1, bias=False)  # number of columns to score
        optimizer = torch.optim.Adam(params=linear_model.parameters())
        loss = ExplodedLogitLoss(loss_type='bce')

        for step in range(dataset_size):
            data, order = next(loader_iterator)
            optimizer.zero_grad()

            score = linear_model(data).squeeze(-1)

            loss_value = loss(score, order)

            l1_loss_value = 0
            for param in linear_model.parameters():
                l1_loss_value += torch.sum(torch.abs(param))
            loss_value += regularization_lambda * l1_loss_value

            loss_value.backward()
            optimizer.step()

            if step % 1000 == 0:
                print("Loss value: {0}".format(loss_value.item()))

        with torch.no_grad():
            for _ in range(test_dataset_size):
                data, expected_order = next(loader_iterator)

                score = linear_model(data).squeeze(-1)
                actual_order = get_sort_order(score)

                self.assertTrue(torch.equal(actual_order, expected_order),
                                "Order not equal:\n{0}\n{1}".format(actual_order, expected_order))

        print("\n\nLinear transformation weights matrix\n--------------------")
        print(linear_model.linear.weight)

        print("\n\nInverted original coefficients\n--------------------")
        print(dataset_generator.coeffs[0])


def get_sort_order(scores):
    s = torch.argsort(scores, descending=True)
    r = torch.zeros(scores.shape, dtype=torch.long)
    for i in range(scores.shape[-1]):
        r[0, s[0, i]] = i
    return r + 1


if __name__ == "__main__":
    unittest.main()
