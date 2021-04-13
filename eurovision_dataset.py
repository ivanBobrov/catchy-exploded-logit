import torch
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
    def __init__(self, dataframe, feature_column_names, shuffle=True):
        self.df = dataframe
        self.f_names = feature_column_names
        self.samples = EurovisionDatasetGrouped.prepare_dataset(self.df, self.f_names, shuffle)

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
                sdf = df[(df.year == year) & (df['round'] == round_type)]

                inputs, targets = EurovisionDatasetGrouped.prepare_sample(sdf, f_cols, shuffle)
                samples.append((inputs, targets))

        return samples

    @staticmethod
    def prepare_sample(sdf, f_cols, shuffle):
        sorted_df = sdf.sort_values('total_points', ascending=False)

        # Generating track's place
        sorted_df.insert(0, 'place', range(1, 1 + len(sorted_df)))

        if shuffle:  # Shuffle back
            sorted_df = sorted_df.sample(frac=1).reset_index(drop=True)

        inputs = torch.tensor(sorted_df[f_cols].values, dtype=torch.float32)
        targets = torch.tensor(sorted_df['place'].values, dtype=torch.long)

        return inputs, targets
