#!/usr/bin/env python3

import numpy as np

def subimage(image, nx, ny):
    """
    Returns a random subimage of an image of size nx x ny
    """
    x = np.random.randint(image.shape[0]-nx)
    y = np.random.randint(image.shape[1]-ny)
    return image[x:x+nx,y:y+ny]

def histogram(image):
    """
    Generates histogram of categorical vairables
    Returns fraction of each category
    """
    categories = np.unique(image)
    histogram = {}
    # Count each category
    for category in categories:
        indicator = [image == category]
        histogram[category] = np.sum(indicator)/image.size

    return histogram

def variogram(image):
    """
    Returns dictionary of indicator variogram values in x direction at all pixels for all categories
    """
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
