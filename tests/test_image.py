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
    img = Image.fromArray(input_data).asArray()
    assert np.alltrue(img == expected)


# ------ Test of conversion functions ------

def test_from_txt():
    img = Image.fromTxt("tests/data/test_img.txt", (3, 3))
    img2 = Image.fromTxt("tests/data/test_img.txt", (3, 3, 1))
    expected = example_image()
    assert img == img2
    assert img == expected


def test_conversion_txt_gslib():
    img = Image.fromTxt("tests/data/test_img.txt", (3, 3))
    img.exportAsGslib("tests/data/test_img.gslib")
    img_test = Image.fromGslib("tests/data/test_img.gslib")
    assert np.alltrue(img_test == img)


def test_gslib_to_vtk():
    img = Image.fromGslib("tests/data/test_img.gslib")
    img.exportAsVtk("tests/data/test_img.vtk")
    img2 = img.fromVtk("tests/data/test_img.vtk")
    assert np.alltrue(img == img2)


def test_vtk_pgm():
    img = Image.fromVtk("tests/data/test_img.vtk")
    img.exportAsPgm("tests/data/test_img.pgm")
    img2 = img.fromPgm("tests/data/test_img.pgm")
    assert np.alltrue(img == img2)


def test_pgm_vox():
    img = Image.fromPgm("tests/data/test_img.pgm")
    img.exportAsVox("tests/data/test_img.vox")
    img_test = Image.fromVox("tests/data/test_img.vox")
    assert np.alltrue(img_test == img)


def test_conversion_vox_png():
    img = Image.fromVox("tests/data/test_img.vox")
    img.exportAsPng("tests/data/test_img.png")
    img_test = Image.fromPng("tests/data/test_img.png")
    assert np.alltrue(img_test == img)


def test_conversion_png_txt():
    img = Image.fromPng("tests/data/test_img.png")
    img.exportAsTxt("tests/data/test_img2.txt")
    img_test = Image.fromTxt("tests/data/test_img2.txt", (3, 3))
    assert np.alltrue(img_test == img)


def test_conversion_final():
    a = np.loadtxt("tests/data/test_img.txt")
    b = np.loadtxt("tests/data/test_img2.txt")
    return np.alltrue(a == b)


def test_import_gslb2var():
    img = Image.fromGslib("tests/data/2var.gslib")
    var1 = img.asArray("V0").reshape((3, 3))
    var2 = img.asArray("V1").reshape((3, 3))
    expected1 = np.array([[1, 1, 1], [1, 1, 1], [0, 0, 0]])
    expected2 = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    assert np.alltrue(var1 == expected1)
    assert np.alltrue(var2 == expected2)


def test_gslib_io():
    img = Image.fromGslib("tests/data/test_color_simple.gslib")
    img.exportAsGslib("tests/data/test_color_simple2.gslib")
    img = Image.fromGslib("tests/data/test_color_simple.gslib")
    img2 = Image.fromGslib("tests/data/test_color_simple2.gslib")
    assert(img == img2)


def test_conversion_color():
    img = Image.fromGslib("tests/data/test_color_simple.gslib")
    img.exportAsPng("tests/data/test_color_simple.png", colored=True)
    img = Image.fromGslib("tests/data/test_color_simple.gslib")
    img.exportAsPpm("tests/data/test_color_simple.ppm")
    img2 = Image.fromPng("tests/data/test_color_simple.png")
    img3 = Image.fromPpm("tests/data/test_color_simple.ppm")
    assert np.alltrue((img.asArray()-img2.asArray()) < 1e8)
    assert img == img3


def test_conversion_color2():
    img = Image.fromGslib("tests/data/test_color.gslib")
    img.exportAsPng("tests/data/test_color.png", colored=True)
    img = Image.fromGslib("tests/data/test_color.gslib")
    img.exportAsPpm("tests/data/test_color.ppm")
    img2 = Image.fromPng("tests/data/test_color.png")
    img3 = Image.fromPpm("tests/data/test_color.ppm")
    assert np.alltrue((img.asArray()-img2.asArray()) < 1e8)
    assert img == img3


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
    assert img == expected


def test_add_variable():
    pass


def test_delete_variable():
    pass
