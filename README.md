# mpstool: Toolbox for Multipoint statistics [WIP]

This is a python3 project providing tools for computing quality indicators for multipoint statistics outputs.
The methods can also be applied to any other 2D images.
It is under construction.

Currently the module provides functions for evaluating connectivity functions of 2D categorical images.

# Installation
Install the mpstool from source (from project main directory):
`pip install .`
or if you want ot run it directly from source (useful for development):
`pip install -e .`

# Example use
```python
import numpy as np
import matplotlib.pyplot as plt
import mpstool

# Load image data stored in a text file and reshape to 2D numpy array
image = np.loadtxt('ti_categoricalSoilCracks.txt').reshape(550, 500)
plt.imshow(image)
plt.show()

# Compute the connectivity function for each category
connectivity = mpstool.connectivity.get_function_x(image)

# Display function for each category on a plot
categories = mpstool.connectivity.get_categories(image)
for category in categories:
    plt.plot(connectivity[category])
plt.legend(categories)
plt.xlabel('distance (pixels)')
plt.ylabel('connectivity')
plt.show()
```

The library also provides a function for retrieving connected components:
```python
# Get the connected components
connected_components = mpstool.connectivity.get_components(image)
plt.imshow(connected_components)
plt.show()
```
