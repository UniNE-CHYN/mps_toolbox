#!/usr/bin/env python3

import numpy as np
import skimage.measure
from mpstool.img import Image
from mpstool.stats import *


def get_function(image, axis, max_lag=None):
    """
    Computes connectivity function along given axis for all categories in image

    Returns a dictionary of connectivity functions.
    Keys of the dictionary are the categories given by get_categories.
    Each entry is a numpy array of connectivity values
    for all pixels in axis  direction for all categories

    Parameters
    ----------
    image : ndarray | Image
    axis : int | axis indicating the direction of the connectivity function
    max_lag : int | maximum lag distance in pixels, default: image.shape[axis]

    Returns
    -------
    dict
        dictionary of connectivity functions,
        each being a numpy array of length max_lag
    """
    if isinstance(image, Image):
        image = image.asArray()

    # Compute connected components and size
    categories = get_categories(image)
    connected_components = get_components(image)

    # Set default maximum lag if not specified
    if max_lag is None:
        nx = image.shape[axis]
    else:
        # quietly use maximum possible value if user input too much
        nx = min(image.shape[axis], max_lag)

    # Compute same categories and same components
    connectivity = {}
    for category in categories:
        mask = np.array(image == category)
        same_category_count = np.zeros(nx-1)
        same_component_count = np.zeros(nx-1)
        for x in np.arange(1, nx):
            same_category_count[x-1] = np.sum(np.logical_and(
                image.take(indices=range(x, nx), axis=axis) == image.take(
                    indices=range(nx-x), axis=axis),
                mask.take(indices=range(x, nx), axis=axis)))
            same_component_count[x-1] = np.sum(np.logical_and(
                connected_components.take(indices=range(x, nx), axis=axis)
                == connected_components.take(indices=range(nx-x), axis=axis),
                mask.take(indices=range(x, nx), axis=axis)))

        # Divide components by categories
        connectivity[category] = np.divide(same_component_count,
                                           same_category_count,
                                           out=np.zeros_like(
                                               same_component_count),
                                           where=same_category_count != 0)

    return connectivity


def get_map(image):
    """
    Computes connectivity map for all categories in image.

    Returns a dictionary of connectivity maps.
    Keys of the dictionary are the categories given by get_categories.
    Each entry is a numpy 2D array of connectivity values for given shifts

    Parameters
    ----------
    image : ndarray | Image
        non-empty numpy array

    Returns
    -------
    ndarray
        2D numpy array of size nx-1, ny-1
    """

    if isinstance(image, Image):
        image = image.asArray()

    # Compute connected components and size
    categories = get_categories(image)
    connected_components = get_components(image)
    nx = image.shape[0]
    ny = image.shape[1]

    # Compute same categories and same components
    connectivity = {}
    for category in categories:
        mask = np.array(image == category)
        same_category_count = np.zeros((nx-1, ny-1))
        same_component_count = np.zeros((nx-1, ny-1))
        for x in np.arange(1, nx):
            for y in np.arange(1, ny):
                same_category_count[x-1, y-1] = np.sum(np.logical_and(
                    image[x:, y:] == image[:-x, :-y], mask[x:, y:]))
                same_component_count[x-1, y-1] = np.sum(np.logical_and(
                    connected_components[x:, y:]
                    == connected_components[:-x, :-y], mask[x:, y:]))
        # Divide components by categories
        connectivity[category] = np.divide(same_component_count,
                                           same_category_count,
                                           out=np.zeros_like(
                                               same_component_count),
                                           where=same_category_count != 0)

    return connectivity


def get_components(image, background=None):
    """
    Computes connected components array of an input image

    Returns array of the same size as the input array.
    The returned array contains integer labels, pixels belonging
    to the same components have the same label.

    http://scikit-image.org/docs/stable/api/skimage.measure.html#label

    Parameters
    ----------
    image : ndarray | Image
        non-empty numpy array

    background : int
        optional argument for background category;
        background pixels will belong to the same connected component
        If no background specified, no background is used at all

    Returns
    -------
    ndarray
        numpy array of the same size as input
    """
    if isinstance(image, Image):
        image = image.asArray()

    # Choose correct background value
    if background is None:
        categories = get_categories(image)
        # empty image should be valid
        if len(categories) > 0:
            # background value which is not present in the image
            background = categories[0]-1

    # Call skimage.measure.label as backend for connected components

    # It appears that it must have some background specified
    # even if we don't want to use any value as background.
    # In this case we set a value not present in the image
    return skimage.measure.label(image, connectivity=1, background=background)
