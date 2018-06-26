#!/usr/bin/env python3

import numpy as np
import skimage.measure

def get_categories(image):
    """
    Find all values in a numpy array

    Return a sorted list of categories found in image 2D arrays.

    Parameters
    ----------
    image : non-empty numpy array

    Returns
    -------
    out : sorted list of all categories (from smallest to greatest)
    """
    # Find categories
    categories = []
    categories.append(image.flatten()[0])
    for pixel in image.flatten():
        if pixel in categories:
            pass
        else:
            categories.append(pixel)

    # Sort categories list
    return np.sort(categories)
    

def get_function(image, axis):
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

def get_function_x(image):
    """
    Computes connectivity function in x direction for all categories in image.

    Returns a dictionary of connectivity functions.
    Keys of the dictionary are the categories given by get_categories.
    Each entry is a numpy array of connectivity values 
    for all pixels in x direction for all categories

    Parameters
    ----------
    image : 2D non-empty numpy array of size nx, ny

    Returns
    -------
    out : dictionary of connectivity functions (numpy arrays of length nx-1) in x for each category
    """
    # Compute connected components and size
    categories = get_categories(image)
    connected_components = get_components(image)
    nx = image.shape[0]

    # Compute same categories and same components
    connectivity = {}
    for category in categories:
        mask = np.array(image == category)
        same_category_count = np.zeros(nx-1)
        same_component_count = np.zeros(nx-1)
        for x in np.arange(1,nx):
            same_category_count[x-1] = np.sum(np.logical_and(image[x:]==image[:-x], mask[x:]))
            same_component_count[x-1] = np.sum(np.logical_and(connected_components[x:]==connected_components[:-x], mask[x:]))
        # Divide components by categories
        connectivity[category] = np.divide(same_component_count, same_category_count, out=np.zeros_like(same_component_count), where=same_category_count!=0)

    return connectivity

def get_function_y(image):
    """
    Computes connectivity function in y direction for all categories in image.

    Returns a dictionary of connectivity functions.
    Keys of the dictionary are the categories given by get_categories.
    Each entry is a numpy array of connectivity values 
    for all pixels in y direction for all categories

    Parameters
    ----------
    image : 2D non-empty numpy array of size nx, ny

    Returns
    -------
    out : dictionary of connectivity functions (numpy arrays of length ny-1) in y for each category
    """
    # Compute connected components and size
    categories = get_categories(image)
    connected_components = get_components(image)
    ny = image.shape[1]

    # Compute same categories and same components
    connectivity = {}
    for category in categories:
        mask = np.array(image == category)
        same_category_count = np.zeros(ny-1)
        same_component_count = np.zeros(ny-1)
        for y in np.arange(1,ny):
            same_category_count[y-1] = np.sum(np.logical_and(image[:,y:]==image[:,:-y], mask[:,y:]))
            same_component_count[y-1] = np.sum(np.logical_and(connected_components[:,y:]==connected_components[:,:-y], mask[:,y:]))
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
    image : 2D non-empty numpy array of size nx, ny

    Returns
    -------
    out : 2D numpy array of size nx-1, ny-1
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

    Parameters
    ----------
    image : 2D non-empty numpy array

    Returns
    -------
    out : numpy array of the same size
    """
    return skimage.measure.label(image, connectivity=1)
