Image module
===================

The image module implements the Image class.
Image class provides conversion primitives and transformation functions to
work on your data.

Conversion
-------------------

Let us create an empty image and export it ::

    import numpy as np
    import mpstool.img as mpsimg

    # creates an image from a numpy array
    image = mpsimg.Image.fromArray(np.zeros((10,10)))

    # creates an image from a txt dump of numpy
    # The shape of the input should be provided here
    image = mpsimg.Image.fromTxt("2D.txt",(550,500))

    # creates an image from a gslib file
    image = mpsimg.Image.fromGslib("2D.gslib")

    # creates an image from a vox file
    image = mpsimg.Image.fromVox("2D.vox")

We can export it to various file formats::

    image.exportAsPng("black.png")
    image.exportAsGslib("black.gslib")
    image.exportAsVox("black.vox")
    image.exportAsTxt("black.txt")

The 3D example is analogous. Be careful however, that some file formats like png
and txt will not work::

    image = mpsimg.Image.fromArray(np.zeros((10,10,10)))
    image.exportAsGslib("black.gslib")
    image.exportAsVox("black.vox")

Normalizing
-----------
Images contain either integer values from 0 to 255 (denormalized) or float numbers
between -1 and 1 (normalized). Methods `.normalize()` and `.unnormalize()` allow you
to switch between the two.

The normalize parameter is set to False by default in the importation functions.


Transformation functions
------------------------
Threshold::

    image.threshold(t=120)
    # values under t are sent to 0. Values above are sent to 255

Saturation of the white::

    image.saturate_white(t=120)
    # Values above t are sent to 255. Values below are unchanged

You can also apply your custom transformation on the values of the pixels::

    f = lambda x : (x+15)%255
    image.apply_fun(f)


Sampling, cutting, tilling
--------------------------
The library also provides functions to extract samples from an image::

    image = mpsimg.Image.fromPng("2D.png")
    sample_shape = (20,20)
    sample = image.get_sample(sample_shape)
    sample.exportAsPng("sample.png")

The cutting function is useful on 3D images. It outputs every cuts of the data along
a given axis. The n parameters tells how many random cuts should be taken. If n equals -1,
all cuts are taken ::

    image = mpsimg.Image.fromTxt("3D.txt",(100,90,80))
    image.exportCuts("cuts_folder",axis=0,n=-1)
    # axis = 0 <-> x ;  1 <-> y ; 2 <-> z

The tilling function takes a list of images and tile them together into a single one::

    image1 = Image.fromGslib("2D.gslib")
    image2 = Image.fromGslib("2D.gslib")
    image3 = Image.fromGslib("2D.gslib")
    image4 = Image.fromGslib("2D.gslib")
    image_list = [image1, image2, image3, image4]
    tiled_horizontal = Image.tile_images(image_list, 'h')
    tiled_vertical = Image.tile_images(image_list, 'v')
    tiled_square = Image.tile_images(image_list, 's')