"""
Test module for stats. Execute with pytest
"""

import numpy as np
import mpstool

import pytest

@pytest.fixture
def image():
    return np.array([[1,2],[2,2],[3,2]])

def test_histogram(image):
    result = {1: 1/6, 2: 4/6, 3: 1/6}
    assert(result == mpstool.stats.histogram(image))

    
