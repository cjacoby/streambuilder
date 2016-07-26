import numpy as np
import os
import pescador
import sklearn
import sklearn.datasets
import tempfile

from nose.tools import raises, eq_

import streambuilder


def get_iris_data():
    # Load up the iris dataset for the demo
    data = sklearn.datasets.load_iris()
    return data.data, data.target


def __test_batch_generation(streamer, max_steps, expected_batch_size,
                            expected_classes=None, class_probs=None):
    """Test a streamer to make sure it generates samples correctly."""
    if expected_batch_size != 1:
        for i in range(max_steps):
            batch = next(streamer)
            assert batch is not None
            assert isinstance(batch, dict)
            assert pescador.batch_length(batch) == expected_batch_size
    else:
        for batch in streamer.generate(max_batches=max_steps):
            assert batch is not None
            assert isinstance(batch, dict)
            assert streambuilder.validate_sample_keys(batch)


def __test_validation_generation(streamer, n_samples, expected_batch_size):
    """A validation generator should generate samples until there are no
    more."""
    pass


# Case 0: Data -> slicer -> streamer -> batches (random sampling)
def test_streambuilder_basic():
    # 0a: (X, y) -> slicer -> streamer -> batches
    X, y = get_iris_data()
    # 0b: [{x_in, target},...] -> slicer -> streamer -> batches
    iris_dataset = streambuilder.to_dict_dataset(X, y)
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
#     iris_dataset = streambuilder.to_dict_dataset(X, y)

#     for batch_size in [1, 10, 100]:
#         streamer = streambuilder.ValidationStreamBuilder(
#             iris_dataset, batch_size=batch_size)
#         yield __test_validation_generation, streamer, len(X), batch_size


# # Case 2: Split data up by class probability
# # Dataset => [Dataset_0, Dataset_i, ..., Dataset_k] => slicers => streamers =>
# #  mux => streamer => batches
# def test_streambuilder_equalclass():
#     X, y = get_iris_data()
#     iris_dataset = streambuilder.to_dict_dataset(X, y)
#     expected_classes = np.unique(y)

#     for batch_size in [1, 10, 100]:
#         for class_probs in ["equal", ((expected_classes + 1) / np.max(
#                 expected_classes + 1))]:
#             streamer = streambuilder.StreamBuilder(
#                 iris_dataset, class_probs=class_probs, batch_size=batch_size)
#             yield __test_batch_generation, streamer, len(X), batch_size, \
#                 expected_classes, class_probs


# # Case 3: Dataset Muxing; Build streams and then combine them with weights.
# def test_mux_datasets():
#     X, y = get_iris_data()
#     X2 = X + np.random.randn(X.shape) * 0.01

#     # 0b: [{x_in, target},...] -> slicer -> streamer -> batches
#     iris_dataset = streambuilder.to_dict_dataset(X, y)
#     iris_dataset_2 = streambuilder.to_dict_dataset(X2, y)

#     ds1 = streambuilder.StreamBuilder(iris_dataset)
#     ds2 = streambuilder.StreamBuilder(iris_dataset_2)

#     max_steps = 50
#     for batch_size in [1, 10, 100]:
#         for weights in [(1, None), (.5, .5), ("equal",)]:
#             streamer = streambuilder.MixDatasets(ds1, ds2,
#                                                  weights=weights,
#                                                  batch_size=batch_size)
#             yield __test_batch_generation, streamer, max_steps, batch_size


# # Case 4: Custom Slicers

# # Other basic functionality:
# def test_test_streamer():
#     def __test(streamer, timed):
#         result = streamer.test(timed=timed)
#         if timed:
#             pass
#         else:
#             assert result is True

#     data = get_iris_data()
#     for timed in [True, False]:
#         streamer = streambuilder.StreamBuilder(*data)
#         yield __test, streamer, timed

