from .img import *


class PointSet:
    """
    Defines a point set:
        npt:     (int) size of the point set (number of points)
        nv:      (int) number of variables (including x, y, z coordinates)
        val:     ((nv,npt) array) attribute(s) / variable(s) values
        varname: (list of string (or string)) variable names
        name:    (string) name of the point set
    """

    def __init__(self,
                 npt=0,
                 nv=0, v=np.nan, varname=None,
                 name=""):
        """
        Inits function for the class

        Parameters
        ----------
        'v' : int/float or tuple/list/ndarray
            value(s) of the new variable:
            if type is int/float: constant variable
            if tuple/list/ndarray: must contain nv*nx*ny*nz values,
                which are put in the image (after reshape if needed)
        """

        self.npt = npt
        self.nv = nv

        # numpy.ndarray (possibly 0-dimensional)
        valarr = np.asarray(v, dtype=float)
        if valarr.size == 1:
            valarr = valarr.flat[0] * np.ones(npt*nv)
        elif valarr.size != npt*nv:
            print('ERROR: v have not an acceptable size')
            return

        self.val = valarr.reshape(nv, npt)

        if varname is None:
            self.varname = []

            if nv > 0:
                self.varname.append("X")

            if nv > 1:
                self.varname.append("Y")

            if nv > 2:
                self.varname.append("Z")

            if nv > 3:
                for i in range(3, nv):
                    self.varname.append("V{:d}".format(i-3))

        else:
            varname = list(np.asarray(varname).reshape(-1))
            if len(varname) != nv:
                print('ERROR: varname have not an acceptable size')
                return

            self.varname = list(np.asarray(varname).reshape(-1))

        self.name = name

    # ------------------------------------------------------------------------
    def set_default_varname(self):
        """Sets default variable names: 'X', 'Y', 'Z', 'V0', 'V1', ..."""

        self.varname = []

        if self.nv > 0:
            self.varname.append("X")

        if self.nv > 1:
            self.varname.append("Y")

        if self.nv > 2:
            self.varname.append("Z")

        if self.nv > 3:
            for i in range(3, self.nv):
                self.varname.append("V{:d}".format(i-3))
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def set_varname(self, vname=None, ind=-1):
        """Sets name of the variable of the given index (if vname is None:
        'V' appended by the variable index is used as vname)."""

        if ind < 0:
            ii = self.nv + ind
        else:
            ii = ind

        if ii < 0 or ii >= self.nv:
            print("Nothing is done! (invalid index)")
            return

        if vname is None:
            vname = "V{:d}".format(ii)
        self.varname[ii] = vname
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def insert_var(self, v=np.nan, vname=None, ind=0):
        """
        Inserts a variable at a given index:

        :param v:   (int/float or tuple/list/ndarray) value(s) of the new
                        variable:
                        if type is int/float: constant variable
                        if tuple/list/ndarray: must contain npt values,
                            which are inserted in the image (after reshape
                            if needed)
        :param vname:   (string or None) name of the insterted variable (set by
                            default if None)
        :param ind: (int) index where the new variable is inserted
        """

        if ind < 0:
            ii = self.nv + ind
        else:
            ii = ind

        if ii < 0 or ii > self.nv:
            print("Nothing is done! (invalid index)")
            return

        # numpy.ndarray (possibly 0-dimensional)
        valarr = np.asarray(v, dtype=float)
        if valarr.size == 1:
            valarr = valarr.flat[0] * np.ones(self.npt)
        elif valarr.size != self.npt:
            print('ERROR: v have not an acceptable size')
            return

        # Extend val
        self.val = np.concatenate((self.val[0:ii, ...],
                                   valarr.reshape(1, self.npt),
                                   self.val[ii:, ...]),
                                  0)
        # Extend varname list
        if vname is None:
            vname = "V{:d}".format(self.nv)
        self.varname.insert(ii, vname)

        # Update nv
        self.nv = self.nv + 1
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def append_var(self, v=np.nan, vname=None):
        """
        Appends one variable:

        :param v:   (int/float or tuple/list/ndarray) value(s) of the new
                        variable:
                        if type is int/float: constant variable
                        if tuple/list/ndarray: must contain npt values,
                            which are appended in the image (after reshape
                            if needed)
        :param vname:   (string or None) name of the appended variable (set by
                            default if None)
        """

        self.insert_var(v=v, vname=vname, ind=self.nv)
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def remove_var(self, ind=-1):
        """Removes one variable (of given index)."""

        if ind < 0:
            ii = self.nv + ind
        else:
            ii = ind

        if ii < 0 or ii >= self.nv:
            print("Nothing is done! (invalid index)")
            return

        # Update val array
        iv = [i for i in range(self.nv)]
        del iv[ii]
        self.val = self.val[iv, ...]

        # Update varname list
        del self.varname[ii]

        # Update nv
        self.nv = self.nv - 1
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def remove_allvar(self):
        """Removes all variables."""

        # Update val array
        self.val = np.zeros((0, self.npt))

        # Update varname list
        self.varname = []

        # Update nv
        self.nv = 0
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def set_var(self, v=np.nan, vname=None, ind=-1):
        """
        Sets one variable (of given index):

        :param v:   (int/float or tuple/list/ndarray) value(s) of the new
                        variable:
                        if type is int/float: constant variable
                        if tuple/list/ndarray: must contain npt values,
                            which are appended in the image (after reshape
                            if needed)
        :param vname:(string) variable name: set only if not None
        :param ind: (int) index of the variable to be set
        """

        if ind < 0:
            ii = self.nv + ind
        else:
            ii = ind

        if ii < 0 or ii >= self.nv:
            print("Nothing is done! (invalid index)")
            return

        # numpy.ndarray (possibly 0-dimensional)
        valarr = np.asarray(v, dtype=float)
        if valarr.size == 1:
            valarr = valarr.flat[0] * np.ones(self.npt)
        elif valarr.size != self.npt:
            print('ERROR: v have not an acceptable size')
            return

        # Set variable of index ii
        self.val[ii, ...] = valarr.reshape(self.npt)

        # Set variable name of index ii
        if vname is not None:
            self.varname[ii] = vname
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def extract_var(self, indlist):
        """Extracts variable(s) (of given index-es)."""

        indlist = [self.nv + i if i < 0 else i for i in indlist]

        if sum([i >= self.nv or i < 0 for i in indlist]) > 0:
            print("Nothing is done! (invalid index list)")
            return

        # Update val array
        self.val = self.val[indlist, ...]

        # Update varname list
        self.varname = [self.varname[i] for i in indlist]

        # Update nv
        self.nv = len(indlist)
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def get_unique(self, ind=0):
        """
        Gets unique values of one variable (of given index):

        :param ind: (int) index of the variable

        :return:    (1-dimensional array) unique values of the variable
        """

        if ind < 0:
            ii = self.nv + ind
        else:
            ii = ind

        if ii < 0 or ii >= self.nv:
            print("Nothing is done! (invalid index)")
            return

        return (np.unique(self.val[ind, ...]))
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def get_prop_one_var(self, ind=0, density=True):
        """
        Gets proportions (density or count) of unique values of one
        variable (of given index):

        :param ind:     (int) index of the variable
        :param density: (bool) computes densities if True and counts otherwise

        :return:    (list (of length 2) of 1-dimensional array) out:
                        out[0]: (1-dimensional array) unique values of
                                the variable
                        out[1]: (1-dimensional array) densities or counts of
                                the unique values
        """

        if ind < 0:
            ii = self.nv + ind
        else:
            ii = ind

        if ii < 0 or ii >= self.nv:
            print("Nothing is done! (invalid index)")
            return

        uv, cv = list(np.unique(self.val[ind, ...], return_counts=True))

        cv = cv[~np.isnan(uv)]
        uv = uv[~np.isnan(uv)]

        if density:
            cv = cv / np.sum(cv)

        return ([uv, cv])
    # ------------------------------------------------------------------------

    def x(self):
        return(self.val[0])

    def y(self):
        return(self.val[1])

    def z(self):
        return(self.val[2])

    def xmin(self):
        return (np.min(self.val[0]))

    def ymin(self):
        return (np.min(self.val[1]))

    def zmin(self):
        return (np.min(self.val[2]))

    def xmax(self):
        return (np.max(self.val[0]))

    def ymax(self):
        return (np.max(self.val[1]))

    def zmax(self):
        return (np.max(self.val[2]))


def readPointSetGslib(filename, missing_value=None):
    """
    Reads a point set from a file (gslib format):

    :param filename:        (string) name of the file
    :param missing_value:   (float or None) value that will be replaced by nan

    :return:                (PointSet class) point set
    """

    # Check if the file exists
    if not os.path.isfile(filename):
        print("Error: invalid filename ({})".format(filename))
        return

    # Open the file in read mode
    with open(filename, 'r') as ff:
        # Read 1st line
        line1 = ff.readline()

        # Read 2nd line
        line2 = ff.readline()

        # Set number of variables
        nv = int(line2)

        # Set variable name (next nv lines)
        varname = [ff.readline().replace("\n", '') for i in range(nv)]

        # Read the rest of the file
        valarr = np.loadtxt(ff)

    # Set number of point(s)
    npt = int(line1)

    # Replace missing_value by np.nan
    if missing_value is not None:
        np.putmask(valarr, valarr == missing_value, np.nan)

    # Set point set
    ps = PointSet(npt=npt, nv=nv, v=valarr.T, varname=varname)

    return (ps)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------


def writePointSetGslib(ps, filename, missing_value=None, fmt="%.10g"):
    """
    Writes a point set in a file (gslib format):

    :param ps:              (PointSet class) point set to be written
    :param filename:        (string) name of the file
    :param missing_value:   (float or None) nan values will be replaced
                                by missing_value before writing
    :param fmt:             (string) single format for variable values, of the
                                form: '%[flag]width[.precision]specifier'
    """

    # Write 1st line in string shead
    shead = "{}\n".format(ps.npt)

    # Append 2nd line
    shead = shead + "{}\n".format(ps.nv)

    # Append variable name(s) (next line(s))
    for s in ps.varname:
        shead = shead + "{}\n".format(s)

    # Replace np.nan by missing_value
    if missing_value is not None:
        np.putmask(ps.val, np.isnan(ps.val), missing_value)

    # Open the file in write binary mode
    with open(filename, 'wb') as ff:
        ff.write(shead.encode())
        # Write variable values
        np.savetxt(ff, ps.val.reshape(ps.nv, -1).T, delimiter=' ', fmt=fmt)

    # Replace missing_value by np.nan (restore)
    if missing_value is not None:
        np.putmask(ps.val, ps.val == missing_value, np.nan)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------


def imageToPointSet(im):
    """
    Returns a point set corresponding to the input image:

    :param im: (Img class) input image

    :return: (PointSet class) point set corresponding to the input image
    """

    # Initialize point set
    ps = PointSet(npt=im.nxyz(), nv=3+im.nv, v=0.0)

    # Set x-coordinate
    t = im.x()
    v = []
    for i in range(im.nyz()):
        v.append(t)

    ps.set_var(v=v, vname='X', ind=0)

    # Set y-coordinate
    t = np.repeat(im.y(), im.nx)
    v = []
    for i in range(im.nz):
        v.append(t)

    ps.set_var(v=v, vname='Y', ind=1)

    # Set z-coordinate
    v = np.repeat(im.z(), im.nxy)
    ps.set_var(v=v, vname='Z', ind=2)

    # Set next variable(s)
    for i in range(im.nv):
        ps.set_var(v=im.val[i, ...], vname=im.varname[i], ind=3+i)

    return (ps)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------


def pointSetToImage(ps, nx, ny, nz, sx, sy, sz, ox, oy, oz, job=0):
    """
    Returns an image corresponding to the input point set and grid:

    :param ps:  (PointSet class) input point set, with x, y, z-coordinates as
                    first three variable
    :param nx, ny, nz: (int) number of grid cells in each direction
    :param sx, sy, sz: (float) cell size in each direction
    :param ox, oy, oz: (float) origin of the grid (bottom-lower-left corner)
    :param job: (int)
                    if 0: an error occurs if one data is located outside of the
                        image grid, otherwise all data are integrated in the
                        image
                    if 1: data located outside of the image grid are ignored
                        (no error occurs), and all data located within the
                        image grid are integrated in the image

    :return: (Img class) image corresponding to the input point set and grid
    """

    if ps.nv < 3:
        print("Error: invalid number of variable (should be > 3)")
        return

    # Initialize image
    im = Img(nx=nx, ny=ny, nz=nz,
             sx=sx, sy=sy, sz=sz,
             ox=ox, oy=oy, oz=oz,
             nv=ps.nv-3, v=np.nan,
             varname=[ps.varname[3+i] for i in range(ps.nv-3)])

    # Get index of point in the image
    xmin, xmax = im.xmin(), im.xmax()
    ymin, ymax = im.ymin(), im.ymax()
    zmin, zmax = im.zmin(), im.zmax()
    ix = np.array(np.floor((ps.val[0]-xmin)/sx), dtype=int)
    iy = np.array(np.floor((ps.val[1]-ymin)/sy), dtype=int)
    iz = np.array(np.floor((ps.val[2]-zmin)/sz), dtype=int)
    # ix = [np.floor((x-xmin)/sx + 0.5) for x in ps.val[0]]
    # iy = [np.floor((y-ymin)/sy + 0.5) for y in ps.val[1]]
    # iz = [np.floor((z-zmin)/sz + 0.5) for z in ps.val[2]]
    for i in range(ps.npt):
        if ix[i] == nx:
            if (ps.val[0, i]-xmin)/sx - nx < 1.e-10:
                ix[i] = nx-1

        if iy[i] == ny:
            if (ps.val[0, i]-ymin)/sy - ny < 1.e-10:
                iy[i] = ny-1

        if iz[i] == nz:
            if (ps.val[0, i]-zmin)/sz - nz < 1.e-10:
                iz[i] = nz-1

    # Check which index is out of the image grid
    # iout = np.any([np.array(ix < 0), np.array(ix >= nx),
    #                np.array(iy < 0), np.array(iy >= ny),
    #                np.array(iz < 0), np.array(iz >= nz)],
    #               0)
    iout = np.any(np.array((ix < 0, ix >= nx,
                            iy < 0, iy >= ny,
                            iz < 0, iz >= nz)), 0)

    if not job and sum(iout) > 0:
        print("Error: point out of the image grid!")
        return

    # Set values in the image
    for i in range(ps.npt):  # ps.npt is equal to iout.size
        if not iout[i]:
            im.val[:, iz[i], iy[i], ix[i]] = ps.val[3:ps.nv, i]

    return (im)
