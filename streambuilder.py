"""
"""

import numpy as np
import pescador
import time


def skl_to_dict_dataset(X, y):
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


def load_npz_dataset(npz_file):
    return np.load(npz_file)['data'].tolist()


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
        assert validate_sample_keys(sample),\
            "{} is missing necessary keys.".format(sample)
        yield sample


class StreamBuilder(object):
    """A Factory class which deals with creating Pescador streammer objects from
    various input types.
    """
    def __init__(self, *sources, batch_size=1):
        self.dataset = []
        self.streamer = None
        self.batch_streamer = None

        self.init_data_source(sources)
        self.init_streamers()
        self.setup_batch(batch_size)

    def init_data_source(self, sources):
        if (len(sources) == 2) and (isinstance(sources[0], np.ndarray) and
                                    isinstance(sources[0], np.ndarray)):
            self.dataset = skl_to_dict_dataset(*sources)
        elif len(sources) == 1 and isinstance(sources[0], str):
            self.dataset = load_npz_dataset(sources[0])
        elif len(sources) == 1 and isinstance(sources[0], list):
            self.dataset = sources[0]
        else:
            raise NotImplementedError("Invalid sources: {}".format(sources))

    def init_streamers(self):
        if self.dataset:
            self.streamer = pescador.Streamer(infinite_dataset_generator,
                                              self.dataset)
        else:
            logger.error("No valid dataset! Fail :(")

    def setup_batch(self, batch_size):
        if self.streamer:
            if batch_size and batch_size > 1:
                self.batch_streamer = pescador.buffer_streamer(
                    self.streamer, batch_size)
        else:
            logger.error("No valid streamer! Fail :(")

    def test(self, timed=False, n_iter=100):
        success = False
        if timed:
            t0 = time.time()

        i = 0
        batch_success = []
        while True:
            batch = next(self)
            batch_success.append(
                (batch is not None and validate_sample_keys(batch)))
            i += 1
            if i >= n_iter:
                break
        success = all(batch_success)

        if timed:
            duration = time.time() - t0
            return dict(duration=duration, success=success)
        else:
            return success

    def __iter__(self):
        if self.batch_streamer:
            return self.batch_streamer
        else:
            return self.streamer.generate()

    def __next__(self):
        stream_source = (self.batch_streamer if self.batch_streamer
                         else self.streamer.generate())
        return next(stream_source)


class ValidationStreamBuilder(StreamBuilder):
    def __new__(cls, *sources, batch_size=1):
        return super(ValidationStreamBuilder, cls).__new__(
            *sources, batch_size)
