# mpstool: Toolbox for Multipoint statistics

This python3 project provides tools for computing quality indicators for multipoint statistics outputs.
The methods can also be applied to 2D or 3D images.

Currently the module provides :
- An Image class, with various import/export/conversion methods to different data types
- Functions for evaluating connectivity, histograms and variograms of 2D and 3D categorical images.
-
[![Build Status](https://travis-ci.org/UniNE-CHYN/mps_toolbox.svg?branch=master)](https://travis-ci.org/UniNE-CHYN/mps_toolbox)
[![Coverage Status](https://coveralls.io/repos/github/UniNE-CHYN/mps_toolbox/badge.svg)](https://coveralls.io/github/UniNE-CHYN/mps_toolbox)
[![Build status](https://ci.appveyor.com/api/projects/status/8guvjt3q2a8xcfde?svg=true)](https://ci.appveyor.com/project/pjuda/mps-toolbox)

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

[![Documentation Status](https://readthedocs.org/projects/mps-toolbox/badge/?version=latest)](https://mps-toolbox.readthedocs.io/en/latest/?badge=latest)
