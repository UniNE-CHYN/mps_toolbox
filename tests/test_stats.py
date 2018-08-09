"""
Test module for connectivity. Execute with pytest.
"""

import numpy as np
import mpstool
import pytest

EPS = 1e-8  # equality threshold for float values


def array():
    return np.array([[1, 1, 1],
                     [0, 0, 0],
                     [1, 0, 1]])


def image():
    return mpstool.img.Image.fromArray(array())


def test_histo():
    expected_histo = {0: 4./9., 1: 5./9.}
    for ar in [array(), image()]:
        assert mpstool.stats.histogram(ar) == expected_histo


def test_vario():
    expected_vario = {0: np.array([0.16666667, 0.]),
                      1: np.array([0.16666667, 0.])}
    for ar in [array(), image()]:
        real_vario = mpstool.stats.variogram(ar)
        assert real_vario.keys() == expected_vario.keys()
        for k in expected_vario.keys():
            assert np.alltrue(abs(real_vario[k]-expected_vario[k]) < EPS)
