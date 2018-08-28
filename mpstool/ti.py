#!/usr/bin/env python3

from mpstool.img import Image
import yaml



class TrainingImageBase:
"""
Implementation of a database of trainig images
"""

    def __init__(self, data, images):
	self._data = data
	self._images = images
