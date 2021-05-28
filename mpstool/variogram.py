
import mpstool.img as mpsimg
import numpy as np


def distmap(shape):
    """
    Computes the distance matrix in number of cells. This is used
    in conjunction the variogram function.
    """
    nl = shape[0]
    nc = shape[1]
    xx, yy = np.meshgrid(np.arange(0, nl), np.arange(0, nc))
    xx = xx - nl / 2
    yy = yy - nc / 2
    dd = np.sqrt(xx**2 + yy**2)
    return dd


def buildindicator_(z):
    """
    Internal function used by fftvariogram() and fftcrossvariogram()

    It builds an indicator function matrix. This matrix contains
    1 everywhere, and 0 for missing values in the input matrix z.
    It also replaces NaN with 0 in the original matrix.

    It returns the indicator, and the corrected input.
    """
    # Builds indicator matrix assuming many 1
    indic = np.ones(z.shape)
    # Locates the missing values (NaN)
    znan = np.isnan(z)
    indic[znan] = 0  # replace NaN by zero
    z[znan] = 0  # replace NaN by zero in original data
    return indic, z


def reduce_matrix_(m, nr, nc, nr2, nc2, nr8, nc8):
    """
    Internal function used by fftvariogram() to reduce the size
    of the matrix m, that was expanded to accelerate the fft.
    Only the relevant data are kept in the final result.
    """
    g = np.zeros((nr2, nc2))
    g[0:nr, 0:nc] = m[0:nr, 0:nc]
    g[0:nr, nc:nc2] = m[0:nr, nc8-nc+1:nc8+1]
    g[nr:nr2, 0:nc] = m[nr8-nr+1:nr8+1, 0:nc]
    g[nr:nr2, nc:nc2] = m[nr8-nr+1:nr8+1, nc8-nc+1:nc8+1]
    g = np.fft.fftshift(g)
    return g


def fftvariogram(input_image):
    """
    Computes the variogram map of a 2D image using Fast Fourier Transform

    Parameters
    ----------

    input_image : numpy array  or  mps_toolbox image
        the input image

    Returns
    -------

    gmap : numpy array
        the variographic map

    nbpairs : numpy array
        the number of pairs

    Examples
    --------

       >>> gmap, npairs = fftvariogram(image)

    Method
    -------

    Since the computation of a variogram map requires to compute
    convolution products, it can be accelerated using Fast Fourier Transform.
    The method was proposed by:

    Marcotte (1996) Fast Variogram Computation with FFT, Computers and
    Geosciences, 22(10): 1175-1186

    This approach is faster than the tradional spatial shift method
    when the size of the image is large enough.

    """

    # Transform the input into a numpy array
    if isinstance(input_image, mpsimg.Image):
        input_image = input_image.asArray()

    # Copy input in array z
    z = input_image.copy()

    # Image size
    nr = z.shape[0]
    nc = z.shape[1]
    if len(z.shape) > 2:
        tmp = z.shape[2]
        if tmp > 1:
            raise ValueError('the input image must be 2D')

    # Ensure proper size before running FFT
    z.shape = (nr, nc,)

    # Variogram map size
    nr2 = 2*nr-1
    nc2 = 2*nc-1

    # Find the closest multiple of 8
    nr8 = int(np.ceil(nr2 / 8) * 8)
    nc8 = int(np.ceil(nc2 / 8) * 8)

    # Matrix of ones (used as an indicator function = 0 for missing values)
    m1, z = buildindicator_(z)

    # Fourrier transforms
    zf = np.fft.fft2(z, (nr8, nc8))
    z2f = np.fft.fft2(z * z, (nr8, nc8))
    m1f = np.fft.fft2(m1, (nr8, nc8))

    # Number of pairs
    npairs = np.round(np.real(np.fft.ifft2(m1f.conjugate() * m1f)))
    npairs = np.maximum(npairs, 1)  # To avoid division by zero

    # Assemble the variogram computation
    tmp = m1f.conjugate() * z2f + z2f.conjugate() * m1f
    tmp -= 2 * zf.conjugate() * zf
    gmap = 0.5 * np.real(np.fft.ifft2(tmp)) / npairs

    # Shift the matrices for readability
    npairs = reduce_matrix_(npairs, nr, nc, nr2, nc2, nr8, nc8)
    gmap = reduce_matrix_(gmap, nr, nc, nr2, nc2, nr8, nc8)

    return gmap, npairs


def vario_error(g1, g2, d, npairs):
    """
    Computes a normalized error between two variogram maps

    The variogram maps are produced for example by the fftvariogram() function.
    They are centered around the zero lag position.

    Parameters
    ----------

    g1 : numpy array
        the first variogram map

    g2 : numpy array
        the second variogram map

    d : numpy array
        distance map, the distance is zero for the central location
        this map can be constructed with the function distmap()

    npairs : numpy array
        map of the number of pairs for each value in the variogram map
        this map is returned by the function fftvariogram()

    Returns
    -------

    error : float
        the weighted sum of error between the two variogram maps


    """

    weight = npairs / d**2
    weight /= np.sum(weight)
    error = np.abs(g1 - g2) * weight
    return np.sum(error)


def fftcrossvariogram(input_image1, input_image2):
    """
    Computes the cross variogram map of 2 2D images using FFT

    Parameters
    ----------

    input_image1 : numpy array  or  mps_toolbox image
        the first input image

    input_image2 : numpy array  or  mps_toolbox image
        the second input image

    Returns
    -------

    gmap1 : numpy array
        the variographic map for the first image

    gmap2 : numpy array
        the variographic map for the second image

    gmap12 : numpy array
        the crossvariogram map for the pair of images

    nbpair1 : numpy array
        the number of pairs for gmap1

    nbpair2 : numpy array
        the number of pairs for gmap2

    nbpair12 : numpy array
        the number of pairs for gmap12


    Examples
    --------

       >>> gmap1, gmap2, gmap12, npair1, npair2, npair12 = fftvariogram(image)

    Method
    -------

    Since the computation of a variogram map requires to compute
    convolution products, it can be accelerated using Fast Fourier Transform.
    The method was proposed by:

    Marcotte (1996) Fast Variogram Computation with FFT, Computers and
    Geosciences, 22(10): 1175-1186

    This approach is faster than the tradional spatial shift method
    when the size of the image is large enough.

    """

    # Transform the input into a numpy array
    if isinstance(input_image1, mpsimg.Image):
        input_image1 = input_image1.asArray()

    # Copy input in array z
    z = input_image1.copy()

    # Transform the input into a numpy array
    if isinstance(input_image2, mpsimg.Image):
        input_image2 = input_image2.asArray()

    # Copy input in array z
    y = input_image2.copy()

    # Image size
    nr = z.shape[0]
    nc = z.shape[1]

    if len(z.shape) > 2:
        tmp = z.shape[2]
        if tmp > 1:
            raise ValueError('the input image must be 2D')

    # Ensure proper size before running FFT
    z.shape = (nr, nc,)
    y.shape = (nr, nc,)

    # Variogram map size
    nr2 = 2*nr-1
    nc2 = 2*nc-1

    # Find the closest multiple of 8
    nr8 = int(np.ceil(nr2 / 8) * 8)
    nc8 = int(np.ceil(nc2 / 8) * 8)

    # Construct matrix of 1 for known values and 0 for missing ones
    idz, z = buildindicator_(z)
    idy, y = buildindicator_(y)

    # Fourrier transforms
    z1f = np.fft.fft2(z, (nr8, nc8))
    z2f = np.fft.fft2(z * z, (nr8, nc8))
    izf = np.fft.fft2(idz, (nr8, nc8))

    y1f = np.fft.fft2(y, (nr8, nc8))
    y2f = np.fft.fft2(y * y, (nr8, nc8))
    iyf = np.fft.fft2(idy, (nr8, nc8))

    # cross-components
    izyf = np.fft.fft2(idz * idy, (nr8, nc8))
    t1 = np.fft.fft2(z * idy, (nr8, nc8))
    t2 = np.fft.fft2(y * idz, (nr8, nc8))
    t12 = np.fft.fft2(z * y, (nr8, nc8))

    # Number of pairs
    npairz = np.round(np.real(np.fft.ifft2(izf.conjugate() * izf)))
    npairz = np.maximum(npairz, 1)  # To avoid division by zero

    npairy = np.round(np.real(np.fft.ifft2(iyf.conjugate() * iyf)))
    npairy = np.maximum(npairy, 1)

    npairzy = np.round(np.real(np.fft.ifft2(izyf.conjugate() * izyf)))
    npairzy = np.maximum(npairzy, 1)

    # Assemble the variogram computation
    tmp = izf.conjugate() * z2f + z2f.conjugate() * izf
    tmp -= 2 * z1f.conjugate() * z1f
    gz = 0.5 * np.real(np.fft.ifft2(tmp)) / npairz

    tmp = iyf.conjugate() * y2f + y2f.conjugate() * iyf
    tmp -= 2 * y1f.conjugate() * y1f
    gy = 0.5 * np.real(np.fft.ifft2(tmp)) / npairy

    tmp = izyf.conjugate() * t12 + t12.conjugate() * izyf
    tmp -= t1.conjugate() * t2 + t2.conjugate() * t1
    gzy = 0.5 * np.real(np.fft.ifft2(tmp)) / npairzy

    # Shift the matrices for readability
    npairz = reduce_matrix_(npairz, nr, nc, nr2, nc2, nr8, nc8)
    npairy = reduce_matrix_(npairy, nr, nc, nr2, nc2, nr8, nc8)
    npairzy = reduce_matrix_(npairzy, nr, nc, nr2, nc2, nr8, nc8)
    gz = reduce_matrix_(gz, nr, nc, nr2, nc2, nr8, nc8)
    gy = reduce_matrix_(gy, nr, nc, nr2, nc2, nr8, nc8)
    gzy = reduce_matrix_(gzy, nr, nc, nr2, nc2, nr8, nc8)

    return gz, gy, gzy, npairz, npairy, npairzy
