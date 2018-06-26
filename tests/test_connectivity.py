"""
Test module for connectivity. Execute with pytest.
"""

import numpy as np
import mpstool

import pytest

@pytest.fixture
def array():
    return  np.array([[1, 1, 1],
                      [0, 0, 0],
                      [1, 0, 1]])

@pytest.fixture
def extruded_array(array):
    return np.array([array, array])

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

def test_connected_component(cube):
    connectivity_array_cube = np.array([[[0, 0, 0],
                                         [0, 0, 0],
                                         [1, 0, 0]],
                                        [[0, 2, 2],
                                         [0, 2, 2],
                                         [0, 0, 0]],
                                        [[0, 2, 2],
                                         [0, 2, 2],
                                         [0, 0, 0]]])
    assert np.alltrue(mpstool.connectivity.get_components(cube) == connectivity_array_cube)

def test_categories(cube):
    assert all(mpstool.connectivity.get_categories(cube) == [0,1])

def test_function_2D(array):
    axis0_connectivity = {0: np.array([1., 0.]), 1: np.array([0., 0.])}
    axis1_connectivity = {0: np.array([1., 1.]), 1: np.array([1. , 0.5])}

    axis0_result = mpstool.connectivity.get_function(array, 0)
    axis1_result = mpstool.connectivity.get_function(array, 1)

    for key in axis0_connectivity:
        assert np.alltrue( axis0_result[key] == axis0_connectivity[key])
    for key in axis1_connectivity:
        assert np.alltrue( axis1_result[key] == axis1_connectivity[key])

def test_function_3D(extruded_array):
    axis0_connectivity = {0: np.array([1.]), 1: np.array([1.])}
    axis1_connectivity = {0: np.array([1., 0.]), 1: np.array([0., 0.])}
    axis2_connectivity = {0: np.array([1., 1.]), 1: np.array([1. ,0.5])}

    axis0_result = mpstool.connectivity.get_function(extruded_array, axis=0)
    axis1_result = mpstool.connectivity.get_function(extruded_array, axis=1)
    axis2_result = mpstool.connectivity.get_function(extruded_array, axis=2)

    for key in axis0_connectivity:
        assert np.alltrue( axis0_result[key] == axis0_connectivity[key])
        assert np.alltrue( axis1_result[key] == axis1_connectivity[key])
        assert np.alltrue( axis2_result[key] == axis2_connectivity[key])
