#!/usr/bin/env python3

import numpy as np

def histogram(image):
    """
    Generates histogram of categorical vairables
    Returns fraction of each category
    """
    if isinstance(image,Image):
        image = image.asArray()
    categories, counts = np.unique(image, return_counts=True)
    return dict(zip(categories, counts/image.size))

def variogram(image):
    """
    Returns dictionary of indicator variogram values in x direction at all pixels for all categories
    """
    if isinstance(image,Image):
        image = image.asArray()

    # Analyse image
    nx = image.shape[0]
    ny = image.shape[1]
    variogram = {}
    categories = np.unique(image)

    # Compute variogram for each category and store in dictionary
    for category in categories:
        mask = np.array(image == category)
        variogram[category] = np.zeros(ny-1)
        for y in np.arange(1, ny):
            variogram[category][y-1] = np.sum(np.logical_and(image[:,y:] != image[:,:-y], mask[:,y:])) / (nx*(ny-y))
    return variogram
