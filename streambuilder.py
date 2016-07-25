"""
"""


def create_dict_dataset(X, y):
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
            'x_in': X[i],
            'target': y[i]
        })
    return dataset
