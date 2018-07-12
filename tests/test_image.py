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

def test_threshold():
    img = example_image()
    img.threshold(t=127)
    thresholded = img.asArray()
    expected = np.array([[255,255,0],
                          [0,0,255],
                          [255,0,0]]).reshape((3,3,1))
    assert expected.shape==thresholded.shape
    assert np.alltrue(thresholded == expected)

def test_saturate():
    img = example_image()
    img.saturate_white(t=127)
    saturated = img.asArray()
    expected = np.array([[255,255,60],
                       [100,10,255],
                       [255,100,0]]).reshape((3,3,1))
    assert saturated.shape==expected.shape
    assert np.alltrue(saturated == expected)

def test_from_list():
    data1 = np.array([[0, 255],
                      [255, 100]])
    data2 =  np.array([[100, 255],
                       [255, 0]])
    input_data = [data1,data2]
    expected = np.array(input_data)
    img = Image.fromArray(input_data)
    assert np.alltrue(img._data == expected)

def test_from_txt():
    img = Image.fromTxt("test_img.txt", (3,3))
    img2 = Image.fromTxt("test_img.txt", (3,3,1))
    expected = example_image()
    assert img == img2
    assert img == expected

def test_conversion_txt_gslib():
    img = Image.fromTxt("test_img.txt", (3,3))
    img.exportAsGslib("test_img.gslib")
    img_test = Image.fromGslib("test_img.gslib")
    assert np.alltrue(img_test._data == img._data)

def test_conversion_gslib_vox():
    img = Image.fromGslib("test_img.gslib")
    img.exportAsVox("test_img.vox")
    img_test = Image.fromVox("test_img.vox")
    assert np.alltrue(img_test._data == img._data)

def test_conversion_vox_png():
    img = Image.fromVox("test_img.vox")
    img.exportAsPng("test_img.png")
    img_test = Image.fromPng("test_img.png")
    assert np.alltrue(img_test._data == img._data)

def test_conversion_png_txt():
    img = Image.fromPng("test_img.png")
    img.exportAsTxt("test_img2.txt")
    img_test = Image.fromTxt("test_img2.txt",(3,3))
    assert np.alltrue(img_test._data == img._data)

def test_conversion_final():
    a = np.loadtxt("test_img.txt")
    b = np.loadtxt("test_img2.txt")
    return np.alltrue(a==b)
