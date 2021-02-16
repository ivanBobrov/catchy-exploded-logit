import os
import sys
import pandas as pd
import numpy as np
import torch
from sklearn import linear_model
from linear_model import LinearModel


def start(feature_directory, dataset_directory):
    print(feature_directory)
    print(dataset_directory)

    dataframe = pd.read_parquet(os.path.join(dataset_directory, "df.parquet"))
    logistic_regression_winner(dataframe)


def linear_regression_pytorch(dataframe):
    x_train, x_test, y_train, y_test = prepare_dataset(dataframe)

    model = LinearModel(x_train.shape[1], 1)

    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10000):
        inputs = torch.from_numpy(x_train).float()
        labels = torch.from_numpy(np.expand_dims(y_train, axis=1)).float()

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print("epoch: " + str(epoch) + ", loss: " + str(loss.item()))

    with torch.no_grad():
        inputs = torch.from_numpy(np.expand_dims(x_test, axis=1)).float()
        predict = model(inputs).numpy().reshape((-1))
        print(predict - y_test)
        print("Average loss: ", (abs(predict - y_test)).mean())


def logistic_regression_winner(dataframe):
    x_train, x_test, y_train, y_test = prepare_dataset(dataframe)
    y_train[y_train > 1] = 0
    y_test[y_test > 1] = 0

    model = LinearModel(x_train.shape[1], 1)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10000):
        inputs = torch.from_numpy(x_train).float()
        labels = torch.from_numpy(np.expand_dims(y_train, axis=1)).float()

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print("epoch: " + str(epoch) + ", loss: " + str(loss.item()))

    with torch.no_grad():
        inputs = torch.from_numpy(np.expand_dims(x_test, axis=1)).float()
        predict = torch.nn.functional.softmax(model(inputs), dim=0).numpy().reshape((-1))
        print(predict)
        print(y_test)
        print("Average loss: ", (abs(predict - y_test)).mean())


def linear_regression_scikit(dataframe):
    x_train, x_test, y_train, y_test = prepare_dataset(dataframe)

    # lr = linear_model.LinearRegression()
    # lr = linear_model.LogisticRegression(penalty='l1', solver='liblinear')
    lr = linear_model.Lasso(alpha=2.2)
    lr.fit(x_train, y_train)

    prediction = lr.predict(x_test)

    loss = abs(prediction - y_test)
    print(loss.mean())


def prepare_dataset(dataframe):
    dataframe.replace([np.inf, -np.inf], np.nan)
    dataframe.fillna(0)
    exclude_columns = ['id', 'path', 'total_points', 'place', 'song.id', 'segment.id',
                       'artist', 'title', 'country', 'to_country', 'year']
    result_column = 'place'

    x_train = dataframe.iloc[:750].drop(exclude_columns, axis=1).values
    y_train = dataframe.iloc[:750][result_column].values

    x_test = dataframe.iloc[750:].drop(exclude_columns, axis=1).values
    y_test = dataframe.iloc[750:][result_column].values

    print('Columns', dataframe.iloc[:750].drop(exclude_columns, axis=1).columns)
    print('Train', x_train.shape, y_train.shape)
    print('Test', x_test.shape, y_test.shape)

    x_train = np.nan_to_num(x_train)
    y_train = np.nan_to_num(y_train)
    x_test = np.nan_to_num(x_test)
    y_test = np.nan_to_num(y_test)

    return (x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "./"
    feature_path = sys.argv[2] if len(sys.argv) > 2 else "./"
    start(feature_path, dataset_path)
