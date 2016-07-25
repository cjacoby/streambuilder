import sklearn
from sklearn.datasets import load_iris

import streambuilder


def iris_data():
    # borrowed from pescador tests.
    data = sklearn.datasets.load_iris()

    return data.data, data.target


def test_dataset():
    def __test_dataset(dataset, X, y):
        assert isinstance(dataset, list)
        assert len(dataset) == len(X) and len(dataset) == len(y)
        for sample in dataset:
            assert 'x_in' in sample and 'target' in sample

    X, y = iris_data()
    dataset = streambuilder.create_dict_dataset(X, y)
    yield __test_dataset, dataset, X, y
