#!/usr/bin/env python3

import numpy as np
import skimage.measure

def categorize(image, thresholds):
    """
    Returns a categorized image according to thresholds specified.

    Parameters
    ----------
    image : ndarray
        non-empty numpy array

    thresholds : array
        must be non-empty and all image values must lie between first and last element of threshold
    
    Returns
    -------
    ndarray
        Array of the same size as the input array, categorized, with labels starting from 1 to n+1,
        where n is the length of the threshold array.
        Category 1 corresponds to the values which are below
        the smallest value in the thresholds array,
        category 2 to values not exceeding the second smallest value, and so forth.
        The last category labels image values which exceed the greatest value found in the thresholds.
        
    """

    # Check if thresholds input is correct
    thresholds_sorted = sorted(thresholds)
    number_of_categories = len(thresholds) + 1

    # Set categories for each interval, starting with the greatest category
    categorized_image = np.ones_like(image, dtype='int') * number_of_categories
    for category in range(number_of_categories-1, 0, -1):
        categorized_image[image <= thresholds_sorted[category-1]] = category

    return categorized_image


def get_categories(image):
    """
    Find all values in a numpy array

    Return a sorted list of categories found in image 2D arrays.

    Parameters
    ----------
    image : ndarray
        non-empty numpy array

    Returns
    -------
    list
        sorted list of all categories (from smallest to greatest)
    """
    # Sort categories list
    return np.unique(image)
    

def get_function(image, axis):
    """
    Computes connectivity function along given axis for all categories in image.

    Returns a dictionary of connectivity functions.
    Keys of the dictionary are the categories given by get_categories.
    Each entry is a numpy array of connectivity values 
    for all pixels in axis  direction for all categories

    Parameters
    ----------
    image : ndarray

    Returns
    -------
    dict
        dictionary of connectivity functions, numpy arrays of length n, where n is lenght of the image along the given axis
    """
    # Compute connected components and size
    categories = get_categories(image)
    connected_components = get_components(image)
    nx = image.shape[axis]

    # Compute same categories and same components
    connectivity = {}
    for category in categories:
        mask = np.array(image == category)
        same_category_count = np.zeros(nx-1)
        same_component_count = np.zeros(nx-1)
        for x in np.arange(1,nx):
            same_category_count[x-1] = np.sum(np.logical_and(image.take(indices=range(x,nx), axis=axis)==image.take(indices=range(nx-x), axis=axis), mask.take(indices=range(x,nx), axis=axis)))
            same_component_count[x-1] = np.sum(np.logical_and(connected_components.take(indices=range(x,nx), axis=axis)==connected_components.take(indices=range(nx-x), axis=axis), mask.take(indices=range(x,nx), axis=axis)))

        # Divide components by categories
        connectivity[category] = np.divide(same_component_count, same_category_count, out=np.zeros_like(same_component_count), where=same_category_count!=0)

    return connectivity

def get_map(image):
    """
    Computes connectivity map for all categories in image.

    Returns a dictionary of connectivity maps.
    Keys of the dictionary are the categories given by get_categories.
    Each entry is a numpy 2D array of connectivity values for given shifts

    Parameters
    ----------
    image : ndarray
        non-empty numpy array

    Returns
    -------
    ndarray
        2D numpy array of size nx-1, ny-1
    """
    # Compute connected components and size
    categories = get_categories(image)
    connected_components = get_components(image)
    nx = image.shape[0]
    ny = image.shape[1]

    # Compute same categories and same components
    connectivity = {}
    for category in categories:
        mask = np.array(image == category)
        same_category_count = np.zeros((nx-1,ny-1))
        same_component_count = np.zeros((nx-1, ny-1))
        for x in np.arange(1,nx):
            for y in np.arange(1,ny):
                same_category_count[x-1, y-1] = np.sum(np.logical_and(image[x:,y:]==image[:-x,:-y], mask[x:,y:]))
                same_component_count[x-1, y-1] = np.sum(np.logical_and(connected_components[x:,y:]==connected_components[:-x,:-y], mask[x:,y:]))
        # Divide components by categories
        connectivity[category] = np.divide(same_component_count, same_category_count, out=np.zeros_like(same_component_count), where=same_category_count!=0)

    return connectivity

def get_components(image):
    """
    Computes connected components array of an input image
    
    Returns array of the same size as the input array.
    The returned array contains integer labels, pixels belonging
    to the same components have the same label.

    http://scikit-image.org/docs/stable/api/skimage.measure.html#label

    Parameters
    ----------
    image : ndarray
        non-empty numpy array

    Returns
    -------
    ndarray
        numpy array of the same size as input
    """
    return skimage.measure.label(image, connectivity=1)


def subimage(image, nx, ny):
    """
    Returns a random subimage of an image of size nx x ny
    """
    x = np.random.randint(image.shape[0]-nx)
    y = np.random.randint(image.shape[1]-ny)
    return image[x:x+nx,y:y+ny]
