#!/usr/bin/env python3
""" Script that converts a one-hot matrix into vector
    of labels
"""


import numpy as np


def one_hot_decode(one_hot):
    """

    Args:
        one_hot: one_hot_encoded matrix

    Returns: np.darray with vector labels

    """
    if not isinstance(one_hot, np.ndarray):
        return None
    elif len(one_hot) == 0:
        return None
    elif len(one_hot.shape) != 2:
        return None
    elif not np.all((one_hot == 0) | (one_hot == 1)):
        return None
    return np.argmax(one_hot, axis=0)
