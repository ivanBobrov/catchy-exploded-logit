import torch
from torch.utils.data import IterableDataset


class ArtificialDataset(IterableDataset):

    def __init__(self, number_of_samples, number_of_competitors,
                 number_of_columns=2, number_of_random_columns=0, rand_eps=1e-3, bias=True):
        self.number_of_samples = number_of_samples
        self.number_of_competitors = number_of_competitors
        self.number_of_columns = number_of_columns
        self.number_of_random_columns = number_of_random_columns
        self.rand_eps = rand_eps

        if bias:
            self.coeffs = torch.randn(number_of_columns).unsqueeze(0).repeat(number_of_competitors, 1)
            self.biases = torch.randn(number_of_columns).unsqueeze(0).repeat(number_of_competitors, 1)
        else:
            self.coeffs = torch.randn(number_of_columns).unsqueeze(0).repeat(number_of_competitors, 1)
            self.biases = torch.zeros(number_of_columns).unsqueeze(0).repeat(number_of_competitors, 1)

    def __iter__(self):
        def dataset_iterator():
            while True:
                order = torch.randperm(self.number_of_competitors) + 1
                target_order = order.unsqueeze(1).repeat(1, self.number_of_columns)
                random_bias = torch.randn((self.number_of_competitors, self.number_of_columns)) * self.rand_eps

                data = self.coeffs * target_order + self.biases + random_bias

                # Adding random columns
                data = torch.cat((data, torch.zeros((self.number_of_competitors, self.number_of_random_columns))), 1)

                yield data, order

        return dataset_iterator()
