import pescador
import sklearn
import sklearn.datasets

from nose.tools import raises, eq_

import streambuilder


def iris_data():
    # Load up the iris dataset for the demo
    data = sklearn.datasets.load_iris()

    return data.data, data.target


def __test_batch_generation(streamer, max_steps, expected_batch_size,
                            expected_classes=None, class_probs=None):
    """Test a streamer to make sure it generates samples correctly."""
    for i in range(max_steps):
        batch = next(streamer)
        assert batch is not None
        assert pescador.batch_length(batch) == expected_batch_size


# Case 0: Data -> slicer -> streamer -> batches (random sampling)
def test_streambuilder_random_sklearn():
    for max_steps in [5, 20, 100]:
        for batch_size in [1, 8, 32, 64]:
            streamer = streambuilder.StreamBuilder.from_skl_data(
                *iris_data(), batch_size=batch_size)
            yield __test_batch_generation, streamer, max_steps, batch_size

            streamer = streambuilder.StreamBuilder.from_dict_data(
                streambuilder.to_dict_dataset(*iris_data()),
                batch_size=batch_size)
            yield __test_batch_generation, streamer, max_steps, batch_size


# 0a: (X, y) -> slicer -> streamer -> batches
# 0b: [{x_in, target},...] -> slicer -> streamer -> batches
# 0c: "file.npz" -> slicer -> streamer -> batches
# 0d: Dataset -> slicer -> streamer -> batches

# Case 1: Data -> slicer -> streamer -> batches (all samples in order / validation)

# Case 2: Split data up by class probability
# Dataset => [Dataset_0, Dataset_i, ..., Dataset_k] => slicers => streamers => mux => streamer => batches

# Case 3: Dataset Muxing; Build streams and then combine them with weights.

# Case 4: Custom Slicers

# Other basic functionality:
# .test()
# .test(timed)
