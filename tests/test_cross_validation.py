"""
Test module for cross validation. Execute with pytest.
"""

import numpy as np
import mpstool
import pytest

@pytest.fixture
def array():
    return np.array([[1, 1, 1],
                     [0, 0, 0],
                     [1, 0, 1]])


def test_sample_random_conditioning_data():

    filename = 'test.gslib'

    ref_data = mpstool.cross_validation.sample_random_conditioning_data(array(), 4)
    mpstool.cross_validation.save_to_gslib(filename, ref_data)
    assert len(ref_data.pixels) == 4

    read_data = mpstool.cross_validation.read_from_gslib(filename)
    assert ref_data == read_data
