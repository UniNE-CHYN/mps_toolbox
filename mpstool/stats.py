#!/usr/bin/env python3

import numpy as np
from mpstool.img import Image


def get_categories(image):
    """
    Find all values in a numpy array

    Return a sorted list of categories found in image 2D arrays.

    Parameters
    ----------
    image : ndarray | Image
        non-empty numpy array

    Returns
    -------
    list
        sorted list of all categories (from smallest to greatest)
    """
    if isinstance(image, Image):
        image = image.asArray()
    return np.unique(image)


def compute_stats(image):
    if isinstance(image, Image):
        image = image.asArray()


def histogram(image):
    """
    Generates histogram of categorical vairables
    Returns fraction of each category
    """
    if isinstance(image, Image):
        image = image.asArray()
    categories, counts = np.unique(image, return_counts=True)
    return dict(zip(categories, counts/image.size))


def variogram(image, axis):
    """
    Returns dictionary of indicator variogram values
    in x direction at all pixels for all categories
    """
    if isinstance(image, Image):
        image = image.asArray()

    # Analyse image
    n = image.shape[axis]
    variogram = {}
    categories = np.unique(image)

    # Compute *volume*
    volume = 1
    for i in range(0, image.ndim):
        volume = volume*image.shape[i]

    # Compute *area*
    area = volume / n

    # Compute variogram for each category and store in dictionary
    for category in categories:
        indicator_image = np.array(image == category)
        variogram[category] = np.zeros(n)
        for x in np.arange(1, n):
            variogram[category][x] = np.sum(
                indicator_image.take(indices=range(x, n), axis=axis) !=
                indicator_image.take(indices=range(n-x), axis=axis)
            ) / (area*(n-x))
    return variogram
