#!/usr/bin/env python3

import numpy as np
import mpstool.cc as cc

def get_categories(image):
    """
    Find all values in a 2D numpy array

    Return a sorted list of categories found in image 2D arrays.

    Parameters
    ----------
    image : 2D non-empty numpy array

    Returns
    -------
    out : sorted list of all categories (from smallest to greatest)
    """
    # Find categories
    categories = []
    categories.append(image[0,0])
    for pixel in image.flatten():
        if pixel in categories:
            pass
        else:
            categories.append(pixel)

    # Sort categories list
    return np.sort(categories)
    
def get_function_x(image):
    """
    Computes connectivity function in x direction for all categories in image.

    Returns a dictionary of connectivity functions.
    Keys of the dictionary are the categories given by get_categories.
    Each entry is a numpy array of connectivity values 
    for all pixels in x direction for all categories

    Parameters
    ----------
    image : 2D non-empty numpy array

    Returns
    -------
    out : sorted list of all categories (from smallest to greatest)
    """
    # Compute connected components and size
    categories = get_categories(image)
    connected_components = get_connected_components(image)
    nx = image.shape[0]

    # Compute same categories and same components
    connectivity = {}
    for category in categories:
        mask = np.array(image == category)
        same_category_count = np.zeros(nx-1)
        same_component_count = np.zeros(nx-1)
        for x in np.arange(1,nx):
            same_category_count[x-1] = np.sum(np.logical_and(image[x:,:]==image[:-x,:], mask[x:,:]))
            same_component_count[x-1] = np.sum(np.logical_and(connected_components[x:,:]==connected_components[:-x,:], mask[x:,:]))
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
    # Initialise and use connected components generator
    cc_generator = cc.connected_components(image)
    cc_generator.fill_label_array()
    return cc_generator.get_label_array()
