"""
Test module for the Image class. Execute with pytest : `pytest test_image.py`
"""

import numpy as np
from mpstool.img import *
import pytest


@pytest.fixture
def example_image():
    data = np.array([[200, 255, 60],
                     [100, 10, 255],
                     [250, 100, 0]])
    return Image.fromArray(data)


def test_threshold1():
    img = example_image()
    img.threshold(thresholds=[127], values=[0, 255])
    expected = Image.fromArray(
        np.array([[255, 255, 0],
                  [0, 0, 255],
                  [255, 0, 0]]))
    assert img == expected


def test_threshold2():
    img = example_image()
    img.threshold(thresholds=[80, 210], values=[0, 127, 255])
    expected = Image.fromArray(
        np.array([[127, 255, 0],
                  [127, 0, 255],
                  [255, 127, 0]]))
    assert img == expected


def test_saturate():
    img = example_image()
    img.saturate(t=127)
    saturated = img.asArray()
    expected = np.array([[200, 255, 0],
                         [0, 0, 255],
                         [250, 0, 0]]).reshape((3, 3, 1))
    assert saturated.shape == expected.shape
    assert np.alltrue(saturated == expected)


def test_labelize():
    img = example_image()
    expected = np.array([[4, 6, 2],
                         [3, 1, 6],
                         [5, 3, 0]])
    labels = labelize(img)
    assert np.alltrue(labels == expected)


def test_from_list():
    data1 = np.array([[0, 255],
                      [255, 100]])
    data2 = np.array([[100, 255],
                      [255, 0]])
    input_data = [data1, data2]
    expected = np.array(input_data)
    img = Image.fromArray(input_data)
    assert np.alltrue(img._data == expected)


def test_from_txt():
    img = Image.fromTxt("tests/test_img.txt", (3, 3))
    img2 = Image.fromTxt("tests/test_img.txt", (3, 3, 1))
    expected = example_image()
    assert img == img2
    assert img == expected


def test_conversion_txt_gslib():
    img = Image.fromTxt("tests/test_img.txt", (3, 3))
    img.exportAsGslib("tests/test_img.gslib")
    img_test = Image.fromGslib("tests/test_img.gslib")
    assert np.alltrue(img_test._data == img._data)


def test_conversion_gslib_vox():
    img = Image.fromGslib("tests/test_img.gslib")
    img.exportAsVox("tests/test_img.vox")
    img_test = Image.fromVox("tests/test_img.vox")
    assert np.alltrue(img_test._data == img._data)


def test_conversion_vox_png():
    img = Image.fromVox("tests/test_img.vox")
    img.exportAsPng("tests/test_img.png")
    img_test = Image.fromPng("tests/test_img.png")
    assert np.alltrue(img_test._data == img._data)


def test_conversion_png_txt():
    img = Image.fromPng("tests/test_img.png")
    img.exportAsTxt("tests/test_img2.txt")
    img_test = Image.fromTxt("tests/test_img2.txt", (3, 3))
    assert np.alltrue(img_test._data == img._data)


def test_conversion_final():
    a = np.loadtxt("tests/test_img.txt")
    b = np.loadtxt("tests/test_img2.txt")
    return np.alltrue(a == b)


def test_categorize1():
    img = example_image()
    expected = Image.fromArray(
        np.array([[255, 255, 100],
                  [100, 100, 255],
                  [255, 100, 100]]))
    print(img._data)
    img.categorize(2)
    print(img._data)
    assert img == expected


def test_categorize2():
    img = example_image()
    expected = Image.fromArray(
        np.array([[255, 255, 100],
                  [100, 0, 255],
                  [255, 100, 0]]))
    img.categorize(3, initial_clusters=[2, 99, 254])
    print(img._data)
    assert img == expected
