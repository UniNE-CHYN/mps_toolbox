"""
Test module for connectivity. Execute with pytest.
"""

import numpy as np
from mpstool.img import Image
import mpstool.connectivity
import pytest


@pytest.fixture
def array():
    return np.array([[1, 1, 1],
                     [0, 0, 0],
                     [1, 0, 1]])


@pytest.fixture
def image(array):
    return Image.fromArray(array)


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


@pytest.fixture
def cube_image(cube):
    return Image.fromArray(cube)


def test_threshold():
    image = Image.fromArray(
        np.array([[0.3, 3.0],
                  [1.2, 2.4],
                  [2.1, 1.1]]))

    categorical_ref = np.array([[0, 2],
                                [1, 2],
                                [2, 1]])

    thresholds = np.array([1.0, 2.0])
    image.threshold(thresholds)
    labels = mpstool.img.labelize(image)
    assert np.alltrue(labels == categorical_ref)


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
    assert np.alltrue(mpstool.connectivity.get_components(
        cube, background=0) == connectivity_array_cube)


def test_categories(cube, cube_image):
    assert all(mpstool.connectivity.get_categories(cube) == [0, 1])
    assert all(mpstool.connectivity.get_categories(cube_image) == [0, 1])


def test_get_map(array, image):
    expected_map = {0: np.array([[1., 0.], [0., 0.]]),
                    1: np.array([[0., 0.], [0., 0.]])}
    for ar in [array, image]:
        real_map = mpstool.connectivity.get_map(ar)
        assert real_map.keys() == expected_map.keys()
        for k in expected_map.keys():
            assert np.alltrue(real_map[k] == expected_map[k])


def test_function_2D(array, image):
    for ar in [array, image]:
        axis0_connectivity = {0: np.array([1., 0.]), 1: np.array([0., 0.])}
        axis1_connectivity = {0: np.array([1., 1.]), 1: np.array([1., 0.5])}

        axis0_result = mpstool.connectivity.get_function(ar, axis=0)
        axis1_result = mpstool.connectivity.get_function(ar, axis=1)

        for key in axis0_connectivity:
            assert np.alltrue(axis0_result[key] == axis0_connectivity[key])
        for key in axis1_connectivity:
            assert np.alltrue(axis1_result[key] == axis1_connectivity[key])


def test_truncated_function_2D(array, image):
    for ar in [array, image]:
        axis0_connectivity = {0: np.array([1.]), 1: np.array([0.])}
        axis1_connectivity = {0: np.array([1.]), 1: np.array([1.])}

        axis0_result = mpstool.connectivity.get_function(ar, axis=0, max_lag=1)
        axis1_result = mpstool.connectivity.get_function(ar, axis=1, max_lag=1)

        for key in axis0_connectivity:
            assert np.alltrue(axis0_result[key] == axis0_connectivity[key])
        for key in axis1_connectivity:
            assert np.alltrue(axis1_result[key] == axis1_connectivity[key])


def test_function_3D(extruded_array):
    axis0_connectivity = {0: np.array([1.]), 1: np.array([1.])}
    axis1_connectivity = {0: np.array([1., 0.]), 1: np.array([0., 0.])}
    axis2_connectivity = {0: np.array([1., 1.]), 1: np.array([1., 0.5])}

    axis0_result = mpstool.connectivity.get_function(extruded_array, axis=0)
    axis1_result = mpstool.connectivity.get_function(extruded_array, axis=1)
    axis2_result = mpstool.connectivity.get_function(extruded_array, axis=2)

    for key in axis0_connectivity:
        assert np.alltrue(axis0_result[key] == axis0_connectivity[key])
        assert np.alltrue(axis1_result[key] == axis1_connectivity[key])
        assert np.alltrue(axis2_result[key] == axis2_connectivity[key])


@pytest.fixture
def c_image():
    return np.array([
        [0.1, 0, 0, 0],
        [0,    0, 0, 0],
        [0,    0, 0.9, 0.9],
        [0,    0, 0.9, 0.9]])


def test_apply_threshold(c_image):
    result1 = np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 0, 0],
        [1, 1, 0, 0]])

    result2 = np.ones_like(c_image)

    assert np.alltrue(
        result1 == mpstool.connectivity._apply_threshold(c_image, 0.2))
    assert np.alltrue(
        result2 == mpstool.connectivity._apply_threshold(c_image, 0.95))


def test_gamma(c_image):
    assert mpstool.connectivity.gamma(c_image, 0) == 0
    assert mpstool.connectivity.gamma(c_image, 0.2) == 1
    assert mpstool.connectivity.gamma(c_image, 0.02) == 1
    assert mpstool.connectivity.gamma(c_image, 0.92) == 1

    assert mpstool.connectivity.gamma(c_image, 0, True) == 1
    assert mpstool.connectivity.gamma(c_image, 0.2, True) == 1
    assert mpstool.connectivity.gamma(c_image, 0.02, True) == 17/25
    assert mpstool.connectivity.gamma(c_image, 0.92, True) == 0


def test_gamma_function(c_image):
    assert np.alltrue(mpstool.connectivity.gamma_function(
        c_image, [0, 0.01, 0.92], True) == [1, 17/25, 0])
