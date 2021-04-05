import unittest
import torch
import pandas as pd

from eurovision_dataset import EurovisionDataset


class EurovisionDatasetTest(unittest.TestCase):

    def test_index(self):
        data = [
            ['final', 2019, 'nl', 'fr', 212, 1., 0.3, 0.4],
            ['semi-final', 2019, 'fr', 'nl', 120, 0.2, 0.7, 0.3],
            ['final', 2020, 'fr', 'de', 300, 0.9, 0.5, 0.2]
        ]

        dataframe = pd.DataFrame(data, columns=['round', 'year', 'from_country_id', 'to_country_id', 'total_points',
                                                'feature1', 'feature2', 'feature3'])

        dataset = EurovisionDataset(dataframe, ['feature1', 'feature2', 'feature3'])

        self.assertEqual(3, len(dataset), "Lengths are not the same")
        for i in range(len(dataset)):
            expected = torch.tensor(data[i][5:]).unsqueeze(0)
            actual_input, actual_order = dataset[i]

            self.assertTrue(torch.equal(expected, actual_input), "Row {0} is not equal".format(i))

    def test_place_order(self):
        data = [
            ['final', 2019, 'nl', 'fr', 212, 1., 0.3, 0.4],
            ['final', 2019, 'nl', 'de', 120, 0.2, 0.7, 0.3],
            ['final', 2019, 'nl', 'pl', 300, 0.9, 0.5, 0.2]
        ]

        dataframe = pd.DataFrame(data, columns=['round', 'year', 'from_country_id', 'to_country_id', 'total_points',
                                                'feature1', 'feature2', 'feature3'])

        dataset = EurovisionDataset(dataframe, ['feature1', 'feature2', 'feature3'], shuffle=False)
        self.assertEqual(1, len(dataset), "Lengths are not the same")

        expected_inputs = torch.tensor([[0.9, 0.5, 0.2],
                                        [1., 0.3, 0.4],
                                        [0.2, 0.7, 0.3]], dtype=torch.float32)
        expected_order = torch.tensor([1, 2, 3], dtype=torch.long)

        actual_inputs, actual_order = dataset[0]
        self.assertTrue(torch.equal(expected_inputs, actual_inputs), "Inputs are not equal")
        self.assertTrue(torch.equal(expected_order, actual_order), "Orders are not equal")

    def test_skip_vote_for_yourself(self):
        data = [
            ['final', 2019, 'nl', 'nl', 212, 1., 1., 1.],
            ['final', 2019, 'nl', 'fr', 276, 2., 2., 2.],
            ['final', 2019, 'fr', 'fr', 120, 3., 3., 3.],
            ['final', 2019, 'fr', 'nl', 192, 4., 4., 4.],
            ['final', 2020, 'pl', 'pl', 300, 5., 5., 5.],
            ['final', 2020, 'pl', 'de', 121, 5., 5., 5.]
        ]

        dataframe = pd.DataFrame(data, columns=['round', 'year', 'from_country_id', 'to_country_id', 'total_points',
                                                'feature1', 'feature2', 'feature3'])

        dataset = EurovisionDataset(dataframe, ['feature1', 'feature2', 'feature3'])

        self.assertEqual(3, len(dataset), "Lengths are not the same")
        for i in range(len(dataset)):
            expected_input = torch.tensor(data[2*i + 1][5:]).unsqueeze(0)
            actual_input, actual_order = dataset[i]

            self.assertEqual(1, actual_input.shape[0])
            self.assertTrue(torch.equal(expected_input, actual_input), "Row {0} is not equal".format(i))
