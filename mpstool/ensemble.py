#!/usr/bin/env python3

import numpy as np
from mpstool.img import Image

class Ensemble():
    """
    Contains list of images and functions to compute
    ensemble averages, etc.
    """
    def __init__(self, image_list):
        self.image_list = image_list


def ensemble_as_subimages(image, subimage_size, ensemble_size):
    image_list = [image.get_sample(subimage_size) for i in range(ensemble_size)]
    return Ensemble(image_list)

def ensemble_from_files(name_generator, open_function, ensemble_size):
    image_list = [open_function(name_generator(i)) for i in range(ensemble_size)]
    return Ensemble(image_list)
