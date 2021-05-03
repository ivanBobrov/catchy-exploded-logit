import unittest
import pandas as pd
import torch
import random

from eurovision_dataset import EurovisionDatasetGrouped


class EurovisionDatasetGroupedTest(unittest.TestCase):

    def test_all_different(self):
        data = [
            ['final', 2019, 'nl', 'fr', 212, 1., 0.3, 0.4],
            ['semi-final', 2019, 'fr', 'nl', 120, 0.2, 0.7, 0.3],
            ['final', 2020, 'fr', 'de', 300, 0.9, 0.5, 0.2]
        ]

        dataframe = pd.DataFrame(data, columns=['round', 'year', 'from_country_id', 'to_country_id', 'total_points',
                                                'feature1', 'feature2', 'feature3'])

        dataset = EurovisionDatasetGrouped(dataframe, ['feature1', 'feature2', 'feature3'],
                                           shuffle_tracks=False, shuffle_batches=False)

        self.assertEqual(3, len(dataset), "Lengths are not the same")
        for i in range(len(dataset)):
            expected = torch.tensor(data[i][5:]).unsqueeze(0)
            actual_input_batch, actual_order_batch = dataset[i]

            self.assertTrue(torch.equal(expected, actual_input_batch[0]), "Row {0} is not equal".format(i))

    def test_years_group(self):
        data = [
            ['final', 2019, 'nl', 'fr', 212, 1., 0.3, 0.4],
            ['final', 2019, 'fr', 'nl', 120, 0.2, 0.7, 0.3],
            ['final', 2020, 'fr', 'de', 300, 0.9, 0.5, 0.2],
            ['final', 2020, 'nl', 'de', 300, 0.9, 0.1, 0.4],
        ]

        dataframe = pd.DataFrame(data, columns=['round', 'year', 'from_country_id', 'to_country_id', 'total_points',
                                                'feature1', 'feature2', 'feature3'])

        dataset = EurovisionDatasetGrouped(dataframe, ['feature1', 'feature2', 'feature3'],
                                           shuffle_tracks=False, shuffle_batches=False)

        self.assertEqual(2, len(dataset), "Lengths are not the same")
        for i in range(len(dataset)):
            for j in range(2):
                expected = torch.tensor(data[2*i+j][5:]).unsqueeze(0)
                actual_input_batch, actual_order_batch = dataset[i]
                self.assertTrue(torch.equal(expected, actual_input_batch[j]), "Row {0} is not equal".format(i))

    def test_judges_group(self):
        random.seed(6)  # fix seed due to shuffle of tracks
        data = [
            ['final', 2019, 'nl', 'fr', 212, 1., 0.3, 0.4],
            ['final', 2019, 'nl', 'de', 120, 0.2, 0.7, 0.3],
            ['final', 2020, 'fr', 'nl', 300, 0.9, 0.5, 0.2],
            ['final', 2020, 'fr', 'de', 300, 0.9, 0.1, 0.4],
        ]

        dataframe = pd.DataFrame(data, columns=['round', 'year', 'from_country_id', 'to_country_id', 'total_points',
                                                'feature1', 'feature2', 'feature3'])

        dataset = EurovisionDatasetGrouped(dataframe, ['feature1', 'feature2', 'feature3'],
                                           shuffle_tracks=False, shuffle_batches=False)

        self.assertEqual(2, len(dataset), "Lengths are not the same")
        for i in range(len(dataset)):
            expected1 = torch.tensor(data[2*i][5:]).unsqueeze(0)
            expected2 = torch.tensor(data[2*i+1][5:]).unsqueeze(0)
            expected = torch.cat((expected1, expected2))
            actual_input_batch, actual_order_batch = dataset[i]
            self.assertTrue(torch.equal(expected, actual_input_batch[0]), "Row {0} is not equal".format(i))
