# StreamBuilder

StreamBuilder is a companion module for Pescador (https://github.com/bmcfee/pescador), intended to shortcut making streams for
datasets for you.

## Use Case 0 - Load a typical streamer from a dataset.
```
import streambuilder
# X, y...
stream = streambuilder.StreamBuilder.from_data(X, y, batch_size=32)
```
or

```
stream = streambuilder.StreamBuilder(dataset, batch_size=32)
```

## Use Case 1 - Equal Class Probabilities
For multiclass datasets, you will typicall have classes with different sizes, a problem for training your learner, where you want to
show an equal number of samples of each class to the learner in each batch. It is easy to fix this with Pescador, multiplexing a stream
of each class, and generating samples from each class with the same probability.

Stream Builder makes this very simple to set up, like this:
```
import streambuilder
# dataset...
stream = streambuilder.StreamBuilder(dataset, equal_class_probs=True, batch_size=32)

batch = next(stream)
# len(batch) == 32
```

You can also force it to cache the stream builder to cache the classes to separate files for faster streaming:
```
stream = streambuilder.StreamBuilder(dataset, equal_class_probs=True, batch_size=32, cache_classes=True)

batch = next(stream)
# len(batch) == 32
```

## Use Case 2 - Mix multiple datasets with the same class data.
If you have several datatasets of the same sort of data, you can mus
```
import streambuilder
# dataset1, dataset2, dataset3
stream = streambuilder.DatasetStreamBuilder(dataset1, dataset2, dataset3, weights='equal', batch_size=32, equal_class_probs=True, cache_classes=True)
# weights = "equal", [.5, .2, .3], etc.
```

## Appendix I - Datasets
Pescador prefers to handle datasets as arrays of dicts, where each dict contains the feature and target information,
instead of separate np.ndarrays of features and targets. StreamBuilder offers a helper to create this:
```
from sklearn.datasets import fetch_mldata
import streambuilder

mnist = fetch_mldata('MNIST original')
dataset = streambuilder.create_dict_dataset(mnist.data, mnist.target)
# 'dataset' is a list of dicts
stream = streambuilder.StreamBuilder(dataset, ...)
```

## Appendix II - Test your stream.
Sometimes you just want to make sure your stream is going to work.

```
stream = ...
success = stream.test()
```

or

```
stream = ...
result = stream.test(timed=True)
print(result['duration'])
print(result['success'])
```
