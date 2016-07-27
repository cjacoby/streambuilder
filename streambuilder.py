"""
"""

from collections import defaultdict
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


class StreamerMixin(object):
    def __init__(self):
        self.source = None
        self.target_datasets = None
        self.streamer = None
        self.batch_streamer = None

    def setup_batch(self, batch_size):
        if self.streamer:
            if batch_size and batch_size > 1:
                self.batch_streamer = pescador.buffer_streamer(
                    self.streamer, batch_size)
        else:
            logger.error("No valid streamer! Failed to setup batches.")

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

    def generate(self, *args, **kwargs):
        return self.streamer.generate(*args, **kwargs)

    def next(self):
        stream_source = (self.batch_streamer if self.batch_streamer
                         else self.streamer.generate())
        return next(stream_source)

    def __next__(self):
        return self.next()


class StreamBuilder(StreamerMixin):
    """A Factory class which deals with creating Pescador streammer objects from
    various input types.
    """
    CLASS_MUX_PARAMS = dict(
        n_samples=None,  # no maximum number
        lam=None,  # Only use these streams; one for each class.
        with_replacement=False,  # really, only these streams
        revive=True  # make sure only one copy of each is active
                     # at a time.
    )

    def __init__(self, *sources,
                 equal_class_probs=None, batch_size=1,
                 cache_class_splits=False):
        """
        Parameters
        ----------
        sources :
            X, y : sklearn-style X, y np.ndarray.
            dict_data : list of dicts
            npz_file : npz filename.

        class_probs : in [None, list of floats, "equal"]
            None : just use the input
            list of floats : Probability of sampling from each
                class. Must match the number of classes in the dataset.
            'equal' : enforce equal probability streaming of each class.

        batch_size : int
            Number of samples to collect for each batch.

        cache_class_splits : bool
            If true, and class_probs is None, after separating
            the different classes into separate sources, also
            caches them to disk so they don't have to be separated again
            next time
        """
        self.source = None
        self.target_datasets = None
        self.streamer = None
        self.batch_streamer = None

        self.init_data_source(sources)
        if equal_class_probs is not None:
            self.separate_target_classes(equal_class_probs, cache_class_splits)
        self.init_seed_pool()
        self.setup_mux()
        self.setup_batch(batch_size)

    def init_data_source(self, sources):
        if (len(sources) == 2) and (isinstance(sources[0], np.ndarray) and
                                    isinstance(sources[0], np.ndarray)):
            self.source = skl_to_dict_dataset(*sources)
        elif len(sources) == 1 and isinstance(sources[0], str):
            self.source = load_npz_dataset(sources[0])
        elif len(sources) == 1 and isinstance(sources[0], list):
            self.source = sources[0]
        else:
            raise NotImplementedError("Invalid sources: {}".format(sources))

    def separate_target_classes(self, equal_class_probs, cache_class_splits):
        self.target_datasets = defaultdict(list)
        self.equal_class_probs = equal_class_probs

        for sample in self.source:
            target = sample.get('target', sample.get('y', sample.get('Y')))
            self.target_datasets[int(target)].append(sample)

        if cache_class_splits:
            raise NotImplementedError("No cache available yet.")

    def init_seed_pool(self):
        if self.target_datasets is not None:
            # Set up target streams
            self.seed_pool = [pescador.Streamer(infinite_dataset_generator,
                                                self.target_datasets[x])
                              for x in self.target_datasets]
        else:
            self.seed_pool = [pescador.Streamer(infinite_dataset_generator,
                                                self.source)]

    def setup_mux(self):
        stream_mux = pescador.mux(self.seed_pool,
                                  k=len(self.seed_pool),
                                  **self.CLASS_MUX_PARAMS)
        self.streamer = pescador.Streamer(stream_mux)


class ValidationStreamBuilder(StreamBuilder):
    CLASS_MUX_PARAMS = dict(
        n_samples=None,
        k=1,
        lam=20000,  # sample ~1000 batches from this stream
        with_replacement=True
    )

    def __init__(self, *sources, batch_size=1, **kwargs):
        return super(ValidationStreamBuilder, self).__init__(
            *sources, batch_size=batch_size, **kwargs)


class StreamMuxer(StreamerMixin):
    """Multiplexes different streams, allowing you
    to mix samples from different datasets together."""

    def __init__(self, *streams, stream_weights=None, batch_size=1):
        """
        Parameters
        ----------
        streams : list of StreamerMixins

        stream_weights : list of float or None
            See pescador.util.mux

        batch_size : int
        """
        super(StreamMuxer, self).__init__()

        self.setup_mux(streams, stream_weights)
        self.setup_batch(batch_size)

    def setup_mux(self, streams, stream_weights):
        stream_mux = pescador.mux(streams,
                                  n_samples=None,
                                  k=len(streams),
                                  lam=None,
                                  pool_weights=stream_weights)
        self.streamer = pescador.Streamer(stream_mux)
