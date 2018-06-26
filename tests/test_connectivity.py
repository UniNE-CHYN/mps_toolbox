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
