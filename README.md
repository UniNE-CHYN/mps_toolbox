# mpstool: Toolbox for Multipoint statistics [WIP]

This is a python3 project providing tools for computing quality indicators for multipoint statistics outputs.
The methods can also be applied to 2D or 3D images.
It is under construction.

Currently the module provides :
- An Image class, with various import/export/conversion methods to different data types
- Functions for evaluating connectivity, histograms and variograms of 2D and 3D categorical images.
-
[![Build Status](https://travis-ci.org/UniNE-CHYN/mps_toolbox.svg?branch=master)](https://travis-ci.org/UniNE-CHYN/mps_toolbox)
[![Coverage Status](https://coveralls.io/repos/github/UniNE-CHYN/mps_toolbox/badge.svg)](https://coveralls.io/github/UniNE-CHYN/mps_toolbox)

# Installation
Install the mpstool from source (from project main directory):
`pip install .`
of use the building script :
`./build.sh`

If you want ot run it directly from source (useful for development):
` pip install -e .`

# Dependencies
- py-vox-io for the .vox file conversion (`pip install py-vox-io`)
- scikit-image (`pip install scikit-image`)
- pillow (`pip install pillow`)

# Documentation

Can be found here: https://mps-toolbox.readthedocs.io/en/latest/index.html

[![Documentation Status](https://readthedocs.org/projects/mps-toolbox/badge/?version=latest)](https://mps-toolbox.readthedocs.io/en/latest/?badge=latest)
