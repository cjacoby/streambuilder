from collections import defaultdict
import numpy as np
import os
import pescador
import sklearn
import sklearn.datasets
import tempfile

from nose.tools import raises, eq_, assert_almost_equals

import streambuilder


def get_iris_data():
    # Load up the iris dataset for the demo
    data = sklearn.datasets.load_iris()
    return data.data, data.target


def __test_batch_generation(streamer, max_steps, expected_batch_size,
                            expected_classes=None, equal_class_probs=None):
    """Test a streamer to make sure it generates samples correctly."""
    class_counts = defaultdict(int)
    total = 0

    for i in range(max_steps):
        batch = next(streamer)
        assert batch is not None
        assert isinstance(batch, dict)
        assert pescador.batch_length(batch) == expected_batch_size

        if equal_class_probs:
            for target in batch['y']:
                class_counts[int(target)] += 1
                total += 1

    if equal_class_probs:
        expected_class_probs = (np.ones(len(expected_classes)) /
                                len(expected_classes))

        for i, expected_prob in enumerate(expected_class_probs):
            actual_prob = class_counts[i] / total

            # Todo: this could be better... and not stochastic
            assert np.abs(expected_prob - actual_prob) < 0.1


def __test_validation_generation(streamer, n_samples, expected_batch_size):
    """A validation generator should generate samples until there are no
    more."""
    pass


# Case 0: Data -> slicer -> streamer -> batches (random sampling)
def test_streambuilder_basic():
    # 0a: (X, y) -> slicer -> streamer -> batches
    X, y = get_iris_data()
    # 0b: [{x_in, target},...] -> slicer -> streamer -> batches
    iris_dataset = streambuilder.skl_to_dict_dataset(X, y)
    tempdir = tempfile.TemporaryDirectory()
    # 0c: "file.npz" -> slicer -> streamer -> batches
    # Set up temporary npz file to load from.
    temp_npz = os.path.join(tempdir.name, "iris.npz")
    np.savez(temp_npz, data=iris_dataset)
    assert os.path.exists(temp_npz)

    # 0d: Dataset -> slicer -> streamer -> batches

    for max_steps in [5, 20, 100]:
        for batch_size in [1, 8, 32, 64]:
            for data_source in [(X, y), (iris_dataset,), (temp_npz,)]:
                streamer = streambuilder.StreamBuilder(
                    *data_source, batch_size=batch_size)
                yield __test_batch_generation, streamer, max_steps, batch_size


# # Case 1: Data -> slicer -> streamer -> batches
# #  (all samples in order / validation)
# def test_streambuilder_validation():
#     X, y = get_iris_data()
#     iris_dataset = streambuilder.skl_to_dict_dataset(X, y)

#     for batch_size in [1, 10, 100]:
#         streamer = streambuilder.ValidationStreamBuilder(
#             iris_dataset, batch_size=batch_size)
#         yield __test_validation_generation, streamer, len(X), batch_size


# Case 2: Split data up by class probability
# Dataset => [Dataset_0, Dataset_i, ..., Dataset_k] => slicers => streamers =>
#  mux => streamer => batches
def test_streambuilder_equalclass():
    X, y = get_iris_data()
    iris_dataset = streambuilder.skl_to_dict_dataset(X, y)
    expected_classes = np.unique(y)

    for batch_size in [1, 10, 100]:
        for equal_class_probs in [True, False]:
            streamer = streambuilder.StreamBuilder(
                iris_dataset, equal_class_probs=equal_class_probs,
                batch_size=batch_size)
            yield __test_batch_generation, streamer, len(X), batch_size, \
                expected_classes, equal_class_probs


# Case 3: Dataset Muxing; Build streams and then combine them with weights.
def test_mux_datasets():
    X, y = get_iris_data()
    X2 = X + np.random.random(X.shape) * 0.01

    # 0b: [{x_in, target},...] -> slicer -> streamer -> batches
    iris_dataset = streambuilder.skl_to_dict_dataset(X, y)
    iris_dataset_2 = streambuilder.skl_to_dict_dataset(X2, y)

    ds1 = streambuilder.StreamBuilder(iris_dataset)
    ds2 = streambuilder.StreamBuilder(iris_dataset_2)

    max_steps = 50
    for batch_size in [1, 10, 100]:
        for weights in [(1.0, 0.0), (.5, .5), (None)]:
            streamer = streambuilder.StreamMuxer(ds1, ds2,
                                                 stream_weights=weights,
                                                 batch_size=batch_size)
            yield __test_batch_generation, streamer, max_steps, batch_size


# Case 4: Custom Slicers

# Other basic functionality:
def test_test_streamer():
    def __test(streamer, timed):
        result = streamer.test(timed=timed)
        if timed:
            assert 'duration' in result and isinstance(
                result['duration'], float)
            assert 'success' in result and result['success']
        else:
            assert result is True

    data = get_iris_data()
    for timed in [True, False]:
        streamer = streambuilder.StreamBuilder(*data)
        yield __test, streamer, timed
