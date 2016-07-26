"""
"""

import numpy as np
import pescador


def to_dict_dataset(X, y):
    """Convert an sklearn style X, y dataset (X=features, y=targets), where
    X and y have the same first dimension, to a list of dictionaries,
    where each dictionary contains the keys 'x_in' and 'target'.

    Parameters
    ----------
    X : np.ndarray

    y : np.ndarray

    Returns
    -------
    dataset : list of dict

    """
    dataset = []
    for i in range(len(X)):
        dataset.append({
            'X': np.atleast_2d(X[i]),
            'y': np.atleast_1d(y[i])
        })
    return dataset


def validate_sample_keys(sample):
    return all([
        'X' in sample or 'x' in sample or 'x_in' in sample,
        'Y' in sample or 'y' in sample or 'target' in sample
    ])


def infinite_dataset_generator(dict_data):
    if not isinstance(dict_data, list):
        raise ValueError("dict_data not appropriately slicable.")
    while True:
        sample = np.random.choice(dict_data)
        assert validate_sample_keys(sample), "{} is missing necessary keys.".format(sample)
        yield sample


class StreamBuilder(object):
    """A Factory class which deals with creating Pescador streammer objects from
    various input types.
    """
    def __new__(cls, *sources, batch_size=1):
        streamer = None
        if (len(sources) == 2) and (isinstance(sources[0], np.ndarray) and
                                    isinstance(sources[0], np.ndarray)):
            streamer = cls.from_skl_data(*sources)
        elif len(sources) == 1 and isinstance(sources[0], str):
            streamer = cls.from_npz(sources[0])
        elif len(sources) == 1 and isinstance(sources[0], list):
            streamer = cls.from_dict_data(sources[0])
        else:
            streamer = cls.from_npz(sources[0])

        if batch_size and batch_size > 1:
            return pescador.buffer_streamer(streamer, batch_size)
        else:
            return streamer

    @classmethod
    def from_skl_data(cls, X, y):
        return cls.from_dict_data(to_dict_dataset(X, y))

    @classmethod
    def from_dict_data(cls, dict_data):
        return pescador.Streamer(infinite_dataset_generator, dict_data)

    @classmethod
    def from_npz(cls, npz_file):
        data = np.load(npz_file)['data'].tolist()
        return cls.from_dict_data(data)


class ValidationStreamBuilder(StreamBuilder):
    def __new__(cls, *sources, batch_size=1):
        return super(ValidationStreamBuilder, cls).__new__(
            *sources, batch_size)
