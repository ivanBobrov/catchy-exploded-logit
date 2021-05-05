import torch
import random
import numpy as np
from torch.utils.data import IterableDataset
from torch.utils.data.dataset import T_co


class EurovisionDataset(IterableDataset):
    def __init__(self, dataframe, feature_column_names, shuffle=True):
        self.df = dataframe
        self.f_names = feature_column_names
        self.samples = EurovisionDataset.prepare_dataset(self.df, self.f_names, shuffle)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index) -> T_co:
        return self.samples[index]

    def __iter__(self):
        return iter(self.samples)

    @staticmethod
    def prepare_dataset(df, f_cols, shuffle):
        samples = []

        years = df.year.unique()
        for year in years:
            year_df = df[df.year == year]

            rounds = year_df['round'].unique()
            for round_type in rounds:
                round_df = df[(df.year == year) & (df['round'] == round_type)]

                judges = round_df.from_country_id.unique()
                for judge in judges:
                    sdf = df[(df.year == year)
                             & (df['round'] == round_type)
                             & (df.from_country_id == judge)]

                    inputs, targets = EurovisionDataset.prepare_sample(sdf, f_cols, shuffle)
                    samples.append((inputs, targets))

        return samples

    @staticmethod
    def prepare_sample(sdf, f_cols, shuffle):
        sorted_df = sdf.sort_values('total_points', ascending=False)

        # Remove voters for yourself
        sorted_df = sorted_df[sorted_df["from_country_id"] != sorted_df["to_country_id"]]

        # Generating track's place
        sorted_df.insert(0, 'place', range(1, 1 + len(sorted_df)))

        if shuffle:  # Shuffle back
            sorted_df = sorted_df.sample(frac=1).reset_index(drop=True)

        inputs = torch.tensor(sorted_df[f_cols].values, dtype=torch.float32)
        targets = torch.tensor(sorted_df['place'].values, dtype=torch.long)

        return inputs, targets


class EurovisionDatasetGrouped(IterableDataset):
    def __init__(self, dataframe, feature_column_names, shuffle_tracks=True, shuffle_batches=True):
        self.df = dataframe
        self.f_names = feature_column_names
        self.shuffle_tracks = shuffle_tracks
        self.shuffle_batches = shuffle_batches
        self.samples = EurovisionDatasetGrouped.prepare_dataset(self.df, self.f_names)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index) -> T_co:
        for i, v in enumerate(self.__iter__()):
            if i == index:
                return v

        return None

    def __iter__(self):
        if self.shuffle_tracks:
            self.shuffle()

        batches = list(self.samples.values())
        if self.shuffle_batches:
            random.shuffle(batches)

        return iter(batches)

    def shuffle(self):
        for year_key, (batch_inputs, batch_targets) in self.samples.items():
            # Shuffling tracks for each judge
            for index, (inputs, targets) in enumerate(zip(batch_inputs, batch_targets)):
                p = np.random.permutation(inputs.shape[0])
                batch_inputs[index, :, :] = inputs[p, :]
                batch_targets[index, :] = targets[p]

            # Shuffling judges inside batches
            p = torch.randperm(batch_inputs.shape[0])
            self.samples[year_key] = (batch_inputs[p, :, :], batch_targets[p, :])

    @staticmethod
    def prepare_dataset(df, f_cols):
        samples = {}

        years = df.year.unique()
        for year in years:
            year_df = df[df.year == year]

            rounds = year_df['round'].unique()
            for round_type in rounds:
                round_df = df[(df.year == year) & (df['round'] == round_type)]

                judges = round_df.from_country_id.unique()
                for judge in judges:
                    sdf = df[(df.year == year)
                             & (df['round'] == round_type)
                             & (df.from_country_id == judge)]

                    inputs, targets = EurovisionDatasetGrouped.prepare_sample(sdf, f_cols)

                    # Separating the case when countries that did not pass
                    # the semi-final still are able to vote for final.
                    # And final number of tracks differ for those judges.
                    postfix = "_sf" if len(round_df.to_country_id.unique()) != inputs.shape[0] + 1 else ""
                    key = str(year) + "_" + round_type + postfix
                    if key not in samples:
                        samples[key] = []

                    samples[key].append((inputs, targets))

        # Preparing batches
        for key in samples.keys():
            list_inputs, list_targets = zip(*samples[key])  # unpack list
            samples[key] = (torch.stack(list_inputs), torch.stack(list_targets))  # reassign as a pair of tensors

        return samples

    @staticmethod
    def prepare_sample(sdf, f_cols):
        sorted_df = sdf.sort_values('total_points', ascending=False)

        # Remove voters for themselves
        sorted_df = sorted_df[sorted_df["from_country_id"] != sorted_df["to_country_id"]]

        # Generating track's place
        sorted_df.insert(0, 'place', range(1, 1 + len(sorted_df)))

        inputs = torch.tensor(sorted_df[f_cols].values, dtype=torch.float32)
        targets = torch.tensor(sorted_df['place'].values, dtype=torch.long)

        return inputs, targets
