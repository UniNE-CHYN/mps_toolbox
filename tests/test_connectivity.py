"""
Test module for connectivity. Execute with pytest.
"""

import numpy as np
import mpstool

import pytest

@pytest.fixture
def cube():
    return np.array([[[0, 0, 0],
                      [0, 0, 0],
                      [1, 0, 0]],
                     [[0, 1, 1],
                      [0, 1, 1],
                      [0, 0, 0]],
                     [[0, 1, 1],
                      [0, 1, 1],
                      [0, 0, 0]]])

def test_categories(cube):
    assert all(mpstool.connectivity.get_categories(cube) == [0,1])

def test_function_x_and_y():
    array = np.array([[1, 1, 1],
                      [0, 0, 0],
                      [1, 0, 1]])
    x_connectivity = {0: np.array([1., 0.]), 1: np.array([0., 0.])}
    y_connectivity = {0: np.array([1., 1.]), 1: np.array([1. , 0.5])}

    x_result = mpstool.connectivity.get_function_x(array)
    y_result = mpstool.connectivity.get_function_y(array)

    for key in x_connectivity:
        assert np.alltrue( x_result[key] == x_connectivity[key])

    for key in y_connectivity:
        assert np.alltrue( y_result[key] == y_connectivity[key])

def test_connected_component(cube):
    connectivity_array = np.array([[[0, 0, 0],
                                     [0, 0, 0],
                                     [1, 0, 0]],
                                    [[0, 2, 2],
                                     [0, 2, 2],
                                     [0, 0, 0]],
                                    [[0, 2, 2],
                                     [0, 2, 2],
                                     [0, 0, 0]]])
    assert np.alltrue(mpstool.connectivity.get_components(cube) == connectivity_array)
