"""
Test module for stats. Execute with pytest
"""

import numpy as np
import mpstool
import pytest


def array():
    return np.array([[1, 1, 1, 1],
                     [0, 0, 0, 0],
                     [1, 0, 1, 1]])


def image():
    return mpstool.img.Image.fromArray(array())


def test_histo():
    expected_histo = {0: 5./12., 1: 7./12.}
    for ar in [array(), image()]:
        assert mpstool.stats.histogram(ar) == expected_histo


def test_vario():
    expected_vario = {0: np.array([0, 2/9, 1/6, 0]),
                      1: np.array([0, 2/9, 1/6, 0])}
    for ar in [array(), image()]:
        real_vario = mpstool.stats.variogram(ar, axis=1)
        assert real_vario.keys() == expected_vario.keys()
        for k in expected_vario.keys():
            assert np.allclose(real_vario[k], expected_vario[k])


def test_vario2():
    expected_vario = {0: np.array([0, 7/8, 1/4]),
                      1: np.array([0, 7/8, 1/4])}
    for ar in [array(), image()]:
        real_vario = mpstool.stats.variogram(ar, axis=0)
        assert real_vario.keys() == expected_vario.keys()
        for k in expected_vario.keys():
            assert np.allclose(real_vario[k], expected_vario[k])
