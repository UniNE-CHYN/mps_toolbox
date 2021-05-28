# mpstool: Toolbox for Multipoint statistics

This python3 project provides tools for computing quality indicators for multipoint statistics outputs.
The methods can also be applied to 2D or 3D images.

Currently the module provides :
- An Image class, with various import/export/conversion methods to different data types
- Functions for evaluating connectivity, histograms and variograms of 2D and 3D categorical images.

[![Documentation Status](https://readthedocs.org/projects/mps-toolbox/badge/?version=latest)](https://mps-toolbox.readthedocs.io/en/latest/?badge=latest)
[![CircleCI](https://circleci.com/gh/UniNE-CHYN/mps_toolbox.svg?style=shield)](https://circleci.com/gh/UniNE-CHYN/mps_toolbox)

# Installation
Install using pip. The package is in the PyPI:
`pip install mpstool`

If you want to run it directly from source, clone this repository and from the root folder run (useful for development):
` pip install -e .`

# Dependencies
- numpy
- py-vox-io
- scikit-image
- pillow
- properscoring

# Documentation

Can be found here: https://mps-toolbox.readthedocs.io/en/latest/index.html

