"""
Test module for the Image class. Execute with pytest : `pytest test_image.py`
"""

import numpy as np
from mpstool.img import *
from copy import copy, deepcopy
import pytest


@pytest.fixture
def img():
    data = np.array([[200, 255, 60],
                     [100, 10, 255],
                     [250, 100, 0]])
    return Image.fromArray(data)


def test_threshold1(img):
    img.threshold(thresholds=[127], values=[0, 255])
    expected = Image.fromArray(
        np.array([[255, 255, 0],
                  [0, 0, 255],
                  [255, 0, 0]]))
    assert img == expected


def test_threshold2(img):
    img.threshold(thresholds=[80, 210], values=[0, 127, 255])
    expected = Image.fromArray(
        np.array([[127, 255, 0],
                  [127, 0, 255],
                  [255, 127, 0]]))
    assert img == expected


def test_saturate(img):
    img.saturate(t=127)
    saturated = img.asArray()
    expected = np.array([[200, 255, 0],
                         [0, 0, 255],
                         [250, 0, 0]]).reshape((3, 3, 1))
    assert saturated.shape == expected.shape
    assert np.alltrue(saturated == expected)


def test_labelize(img):
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

def test_from_txt(img):
    img1 = Image.fromTxt("tests/data/test_img.txt", (3, 3))
    img2 = Image.fromTxt("tests/data/test_img.txt", (3, 3, 1))
    assert img == img1
    assert img == img2


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
    a = np.loadtxt("tests/data/test_img.txt").reshape((3, 3))
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


def test_categorize1(img):
    expected = Image.fromArray(
        np.array([[255, 255, 100],
                  [100, 100, 255],
                  [255, 100, 100]]))
    print(img._data)
    img.categorize(2)
    print(img._data)
    assert img == expected


def test_categorize2(img):
    expected = Image.fromArray(
        np.array([[255, 255, 100],
                  [100, 0, 255],
                  [255, 100, 0]]))
    img.categorize(3, initial_clusters=[2, 99, 254])
    assert img == expected


def test_dimension(img):
    assert img.nxyz() == 9
    assert img.nxy() == 9
    assert img.nxz() == 3
    assert img.nyz() == 3
    assert img.xmin() == 0
    assert img.xmax() == 3
    assert img.ymin() == 0
    assert img.ymax() == 3
    assert img.zmin() == 0
    assert img.zmax() == 1
    coords = np.array([0.5, 1.5, 2.5])
    assert np.alltrue(img.x() == coords)
    assert np.alltrue(img.y() == coords)
    assert np.alltrue(img.z() == np.array([0.5]))
    assert img.vmin() == 0
    assert img.vmax() == 255
    assert img.get_variables() == ["V0"]


def test_variable(img):
    data = np.array([[200, 255, 60],
                     [100, 10, 255],
                     [250, 100, 0]])
    img.add_variable("test", data)
    assert set(img.get_variables()) == {"V0", "test"}
    assert np.alltrue(img._data["test"] == data)
    data = np.zeros(img.shape)
    img.set_variable("test", data)
    assert np.alltrue(img._data["test"] == data)
    img.rename_variable("test", "toto")
    assert np.alltrue(img._data["toto"] == data)
    img.remove_variable("toto")
    assert img.get_variables() == ["V0"]


def test_reset_default_variable_name(img):
    data = np.array([[200, 255, 60],
                     [100, 10, 255],
                     [250, 100, 0]])
    img.add_variable("custom_var_name", data)
    img.reset_var_names_to_default()
    # default variable names are V0, V1... V<n_variables>
    assert img.get_variables() == ["V0", "V1"]


def test_extract(img):
    img2 = deepcopy(img).extract_variable(["V0"], copy=False)
    assert len(img2.get_variables()) == 1
    assert img2 == img


def test_flip(img):
    img2 = img
    img2.flipx()
    img2.flipy()
    img2.flipz()
    img2.flipx()
    img2.flipy()
    img2.flipz()
    assert img2 == img


def test_perm(img):
    img2 = img
    img2.permxy()
    img2.permyz()
    img2.permxz()
    img2.permyz()
    assert img == img2
