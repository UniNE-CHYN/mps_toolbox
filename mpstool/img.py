#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
file:           img.py
author:         Guillaume Coiffier, Julien Straubhaar
last update:    September 2018

Definition of class Image, and relative functions.
"""

import numpy as np
import os
from math import sqrt


class UndefVarExc(Exception):
    def __init__(self, fun_name):
        Exception.__init__(self, "Arguments of method {} are ambigous : \
                                several variables were found.\
                                Please specify a var_name \
                                to work with.".format(fun_name))


class Image:
    """
    A data container. Defines an image as a 3D grid
    with variable(s) / attribute(s):
     shape:    (3-uple of int) number of grid cells in each direction
     spacing : (3-uple of  float) cell size in each direction
     origin :  (3-uple  float) origin of the grid (bottom-lower-left corner)
     nv:       (int) number of variable(s) / attribute(s)
     data:     dictionnary of (nz,ny,nx) arrays
     name:     (string) name of the image

    The underlying data structure is a numpy ndarray.
    Implements transformations on images, as well as type conversion
    from the following data types :

    *png*
    *pgm*
    *ppm*
    *vtk*

    *gslib*
    The .gslib format is a text format with the following structure :

    | nx  ny  nz  x_orig  y_orig  z_orig x_spacing y_spacing z_spacing
    | number_of_variables (should be 1 for our use)
    | name_of_variable_1
    | ...
    | name_of_variable_n

    then, on each line, the value of a coordinate (x,y,z), in the order of the
    nested loop :

    for x = 1 to nx
        for y = 1 to ny:
            for z = 1 to nz:

    *vox*
    A binary file used in MagicaVoxel and in other voxel editors.
    https://ephtracy.github.io/index.html?page=mv_main
    """

    def __init__(self, data, params):
        """
        Initialisation method.
        It is not meant to be called.
        Instead, use methods as fromGslib, fromArray, fromPng, ...
        """
        self._data = data
        self.is3D = params["is3D"]
        self.nvariables = 1

        if "nVariables" in params:
            self.nvariables = len(data)

        self.orig = (0, 0, 0)
        if "origin" in params:
            self.orig = params["origin"]

        self.spacing = (1, 1, 1)
        if "spacing" in params:
            self.spacing = params["spacing"]

        example_data = list(self._data.values())[0]
        if self.is3D:
            self.shape = example_data.shape
        else:
            self.shape = example_data.shape[:3]
            if len(example_data.shape) == 2:
                self.shape = self.shape + (1,)
                for key in self._data:
                    self._data[key] = self._data[key].reshape(self.shape)
            elif len(example_data.shape) == 3 and example_data.shape[2] == 4:
                # get rid of transparency channel
                for key in self._data:
                    self._data[key] = self._data[key][:, :, :3]

    def __eq__(self, other):
        if self._data.keys() != other._data.keys():
            return False
        for k in self._data.keys():
            if self._data[k].shape != other._data[k].shape:
                return False
            if not np.alltrue(self._data[k] == other._data[k]):
                return False
        return True

    def __str__(self):
        return self._data.__str__()

    # ---------------- Import/Export methods ----------------------------------

    # ------ Numpy Array ------

    @staticmethod
    def fromArray(ar, var_first=True, normalize=False):
        """
        Image staticmethod. Used as an initializer.
        Builds the container from a numpy array

        Parameters
        ----------
        'ar' : ndarray | list of ndarray
            The numpy array around which the Image object is built
            A list of 2D ndarray can be given in order to build a 3D image

        'var_first' : boolean
            Indicates if the variable index is the first of the last
            of the array. Set to True by default.

        Returns
        ----------
        A new Image object
        """
        if isinstance(ar, list):
            ar = np.array(ar)

        data = {}
        if len(ar.shape) > 3:
            if var_first:
                for i in range(ar.shape[0]):
                    data["V{}".format(i)] = ar[i, ...]
            else:
                for i in range(ar.shape[3]):
                    data["V{}".format(i)] = ar[:, :, :, i]
        else:
            data["V0"] = ar
        shape = ar.shape
        params = dict()
        if len(shape) < 3 or shape[2] == 1:
            params["is3D"] = False
        else:
            params["is3D"] = True
        output = Image(data, params)
        if normalize:
            output.normalize()
        return output

    @staticmethod
    def empty(size, default_value=np.nan):
        img = Image.fromArray(np.full(size, default_value))
        img.remove_variable("V0")
        return img

    def asArray(self, var_name=None):
        """
        Return the raw data as a numpy array

        Parameters
        ----------
        'var_name' : string
            The variable to export. If None, all variables will be exported and
            the variable selection will be the first dimension of the array
        """
        if self.nvariables == 1:
            return list(self._data.values())[0]
        if var_name is None:
            return np.array(list(self._data.values()))
        else:
            return self._data[var_name]

    # ------ GSLIB ------
    @staticmethod
    def fromGslib(file_name: str, normalize=False, missing_value=None):
        """
        Image staticmethod. Used as an initializer.
        Builds the container from a .gslib file.

        Parameters
        ----------
        'file_name' : string
            relative path to the gslib file

        'normalize' : boolean
            if set to true, the values will be stretched to fit in [-1;1]

        Returns
        ----------
            A new Image object
        """
        with open(file_name, 'r') as f:
            params = dict()
            metadata = f.readline().strip().split()
            xdim = int(metadata[0])
            ydim = int(metadata[1])
            zdim = int(metadata[2])
            params["is3D"] = (zdim > 1)
            if len(metadata) > 3:
                xstep = float(metadata[3])
                ystep = float(metadata[4])
                zstep = float(metadata[5])
                params["spacing"] = (xstep, ystep, zstep)
                xorig = float(metadata[6])
                yorig = float(metadata[7])
                zorig = float(metadata[8])
                params["origin"] = (xorig, yorig, zorig)
            nVar = int(f.readline().strip())
            params["nVariables"] = nVar
            var_names = []
            for i in range(nVar):
                var_names.append(f.readline().strip())
            data = dict([(var_names[i], np.zeros((zdim, ydim, xdim)))
                         for i in range(nVar)])
            for iz in range(zdim):
                for iy in range(ydim):
                    for ix in range(xdim):
                        values = f.readline().strip().split()
                        for iv in range(nVar):
                            key = var_names[iv]
                            data[key][iz, iy, ix] = np.float32(values[iv])
        for k in data:
            data[k] = data[k].T
            if missing_value is not None:
                # Replace missing_value by np.nan
                np.putmask(data[k], data[k] == missing_value, np.nan)
        img = Image(data, params)
        if normalize:
            img.normalize()
        return img

    def exportAsGslib(self, output_name: str, verbose=False):
        """
        Export the Image object data as a gslib file.

        Parameters
        ----------
        'output_name' : string
            relative path to the gslib file to be output
        'verbose' : boolean
            enables verbose mode. Set to False by default
        """
        with open(output_name, 'w') as f:
            xdim = self.shape[0]
            ydim = self.shape[1]
            if self.is3D:
                zdim = self.shape[2]
            else:
                zdim = 1
            xs, ys, zs = self.spacing
            xo, yo, zo = self.orig
            f.write("{} {} {} {} {} {} {} {} {}\n".format(
                xdim, ydim, zdim, xs, ys, zs, xo, yo, zo))
            f.write(str(self.nvariables)+"\n")
            for key in self._data:
                f.write(key+"\n")
            for iz in range(zdim):
                for iy in range(ydim):
                    for ix in range(xdim):
                        toWrite = ""
                        for key in self._data:
                            toWrite += str(self._data[key][ix, iy, iz])
                            toWrite += " "
                        f.write(toWrite+"\n")
        if verbose:
            print("Generated .gslib file as {}".format(output_name))

    # ------ Txt (numpy dump) ------

    @staticmethod
    def fromTxt(file_name: str, shape, normalize=False):
        """
        Image staticmethod. Used as an initializer.
        Builds the container from a raw txt file

        Parameters
        ----------
        'file_name' : string
            the relative path to the txt file to read
        'shape' : tuple of int
            the shape of the data (shape is not contained in the raw txt file,
            thus this parameter is necessary)

        Returns
        ----------
        A new Image object
        """
        array = np.loadtxt(file_name).reshape(shape)
        return Image.fromArray(array, normalize)

    def exportAsTxt(self, output_name: str, verbose=False):
        """
        Export the Image object data as a txt file.
        Requires the data to be two dimensionnal.

        Parameters
        ----------
        'output_name' : string
            relative path to the txt file to be output
        'verbose' : boolean
            enables verbose mode. Set to False by default
        """
        if ".txt" in output_name:
            output_name = output_name.split(".txt")[0]

        if self.is3D:
            raise Exception("ERROR : Export as a txt file requires the data \
                   to be 2 dimensionnal.")
        for key in self._data:
            output = self._data[key].reshape(self.shape[:2])
            final_output_name = output_name+"_"+key + \
                ".txt" if key != "V0" else output_name+".txt"
            np.savetxt(final_output_name, output)
        if verbose:
            print("Generated txt file as {}".format(output_name))

    # ------ Png ------

    @staticmethod
    def fromPng(file_name: str, normalize=False, channel_mode="RGB"):
        """
        Image staticmethod. Used as an initializer.
        Builds the container from a .png file.
        If the png file is colored, three variables, named R,G and B will
        be imported.

        Makes calls to the Pillow library

        Parameters
        ----------
        'file_name' : string
            relative path to the png file

        'normalize' : boolean
            if set to true, the values will be stretched to fit in [-1;1]

        'channel_mode' : string
            if set to RGB, will extract three variables R,G and B
            if set to Gray, will perform a grayscale transformation and
            extract only one variable.
            Default is RGB

        Returns
        ----------
        A new Image object
        """
        try:
            from PIL import Image as PIL_Img
        except ImportError:
            print("Cannot read from png. Is the pillow library installed ?\n\
                   To install it, run `pip install pillow`")
            return
        ar = PIL_Img.open(file_name)
        if ar.mode == 'P':
            ar = ar.convert('RGB')
        ar = np.array(ar).astype(np.float32)
        shape = ar.shape
        if len(shape) == 2 or shape[2] == 1:  # only one canal
            data = {"V0": ar}
            params = {"is3D": False, "nVariables": 1}
        else:
            if channel_mode == "RGB":
                data = {"R": ar[:, :, 0],
                        "G": ar[:, :, 1],
                        "B": ar[:, :, 2]}
                # Ignore the eventual alpha canal
                params = {"is3D": False, "nVariables": 3}
            elif channel_mode == "Gray":
                gray_ar = 0.29*ar[:, :, 0] + 0.58 * \
                    ar[:, :, 1] + 0.13*ar[:, :, 2]
                data = {"V0": gray_ar}
                params = {"is3D": False, "nVariables": 1}
        img = Image(data, params)
        if normalize:
            img.normalize()
        return img

    def exportAsPng(self, output_name: str, colored=True,
                    color_channels=['R', 'G', 'B'], verbose=False):
        """
        Export the Image object data as a png file.
        Requires the data to be two dimensionnal.

        Parameters
        ----------
        'output_name' : string
            relative path to the png file to be output

        'colored' : boolean
            Say if the output images is colored.
            If colored is False, will output every variable as a different png.
            If colored is True, will output one png file with 3 channels.
            Channels are defined in the color_channels argument

        'color_channels' : list of 3 strings
            The name of the three variables to be taken as red, green and
            blue channels. Default names are ['R', 'G', 'B'].
            This argument is ignored if colored is False

        'verbose' : boolean
            enables verbose mode. Set to False by default
        """
        try:
            from PIL import Image as PIL_Img
        except Exception as e:
            print("ERROR : Cannot export as a png. \
                   Received the following error :\n{}\n\
                   Is the pillow library installed ?\n\
                   To install it, run `pip install pillow`".format(e))
            return

        if self.is3D:
            raise Exception("ERROR : Export as a png file requires the data \
                                to be 2 dimensionnal.")
        if self.nvariables == 1:
            colored = False
        if colored:
            assert len(color_channels) == 3
            self.unnormalize(color_channels)
            r_data, g_data, b_data = (self._data[k] for k in color_channels)
            data = np.array([r_data, g_data, b_data])
            data = np.moveaxis(data, 0, -1)
            output = PIL_Img.fromarray(np.squeeze(data))
            output.save(output_name, mode="RGB")
            if verbose:
                print("Generated image as {}".format(output_name))
        else:
            if ".png" in output_name:
                output_name = output_name.split(".png")[0]
            self.unnormalize()
            for key in self._data:
                output_k = PIL_Img.fromarray(np.squeeze(self._data[key]))
                final_output_name = output_name+"_" + key + ".png"\
                    if key != "V0" else output_name+".png"
                if verbose:
                    print("Generated image as {}".format(final_output_name))
                output_k.save(final_output_name, mode="L")  # L for greyscale

    # ------- Vox ------

    @staticmethod
    def fromVox(file_name: str):
        """
        Image staticmethod. Used as an initializer.
        Builds the container from a .vox file.
        Requires py-vox-io to function

        Parameters
        ----------
        'file_name' : string
            relative path to the gslib file

        Returns
        ----------
        A new Image object
        """
        try:
            from pyvox.models import Vox
            from pyvox.parser import VoxParser
        except ImportError:
            print("py-vox-io is not installed. Cannot import a vox file.\n\
                  Please install py-vox-io with `pip install py-vox-io`")
            return
        from pyvox.writer import VoxWriter
        ar = VoxParser(file_name).parse().to_dense()
        data = {"V0": ar}
        params = dict([("is3D", ar.shape[2] > 1)])
        return Image(data, params)

    def exportAsVox(self, output_name: str, verbose=False):
        """
        Export the Image object data as a vox file.
        Requires the image to be a black and white one (only one channel).
        Requires py-vox-io to function.

        Parameters
        ----------
        'output_name' : string
            relative path to the vox file to be output
        'verbose' : boolean
            enables verbose mode.
            default=False
        """
        try:
            from pyvox.models import Vox
            from pyvox.writer import VoxWriter
        except ImportError:
            print("py-vox-io is not installed. Cannot export as vox file.\n\
                  Please install py-vox-io with `pip install py-vox-io`")
            return
        self.unnormalize()

        # Crop to 255, otherwise conversion fails because input is too big
        # (Dimension has to fit in uint8 data type)
        if ((np.array(self.shape) > 255).any()):
            print("[WARNING] the image shape {} is to big to be converted \
                  into vox.\n It will be cropped at 255.")
        for key in self._data:
            a = self._data[key].copy()
            a = a[:255, :255, :255]
            vox = Vox.from_dense(a)
            final_output_name = output_name+"_"+key if key != "V0" \
                else output_name
            VoxWriter(final_output_name, vox).write()
        if verbose:
            print("Generated .vox file as {}".format(output_name))

    # ------ Vtk ------
    @staticmethod
    def fromVtk(file_name: str, missing_value=None):
        """
        Image static method. Used as an initializer.
        Builds the container from a .vtk file.

        Parameters
        ----------
        'file_name' : string
            name of the file

        'missing_value' : float|None
            value that will be replaced by nan

        Returns
        ----------
        A new Image object
        """

        # Check if the file exists
        if not os.path.isfile(file_name):
            raise Exception("Error: invalid filename ({})".format(filename))

        # Open the file in read mode
        with open(file_name, 'r') as ff:
            # Read lines 1 to 10
            header = [ff.readline() for i in range(10)]
            # Read the rest of the file
            val_arr = np.loadtxt(ff)

        # Set grid
        shape = [int(n) for n in header[4].split()[1:4]]
        orig = [float(n) for n in header[5].split()[1:4]]
        spacing = [float(n) for n in header[6].split()[1:4]]

        # Set variable
        tmp = header[8].split()
        nvariables = int(tmp[3])
        var_name = tmp[1]

        # Replace missing_value by np.nan
        if missing_value is not None:
            np.putmask(val_arr, val_arr == missing_value, np.nan)

        # create image
        data = {var_name: val_arr.reshape(shape)}
        params = {
            "origin": orig,
            "spacing": spacing,
            "is3D": shape[2] > 1,
            "nVariables": nvariables
        }
        return Image(data, params)

    def exportAsVtk(self, output_name: str, var_name=None, missing_value=None,
                    fmt="%.10g", data_type='float', version=3.4):
        """
        Export the Image object data as a vtk file.

        Parameters
        ----------
        'output_name' : string
            relative path to the vox file to be output

        'var_name' : string
            The variable to be exported.
            name to be written at line 2

        'verbose' : boolean
            enables verbose mode.
            default=False

        'missing_value' : float|None
            nan values will be replaced by this value before writing

        'fmt' : string
            single format for variable values, of the form:
                '%[flag]width[.precision]specifier'

        'data_type' : string
            data type (can be 'float', 'int', ...)

        'version' : float
            version number (for data file)
        """

        if var_name is None and self.nvariables > 1:
            raise UndefVarExc("exportAsVtk")
        key = var_name if var_name is not None else self.get_variables()[0]
        data = self._data[key]

        nx, ny, nz = self.shape
        ox, oy, oz = self.orig
        sx, sy, sz = self.spacing

        # Set header (10 first lines)
        shead = (
            "# vtk DataFile Version {0}\n"
            "{1}\n"
            "ASCII\n"
            "DATASET STRUCTURED_POINTS\n"
            "DIMENSIONS {2} {3} {4}\n"
            "ORIGIN     {5} {6} {7}\n"
            "SPACING    {8} {9} {10}\n"
            "POINT_DATA {11}\n"
            "SCALARS {12} {13} {14}\n"
            "LOOKUP_TABLE default\n"
        ).format(version,
                 key,
                 nx, ny, nz,
                 ox, oy, oz,
                 sx, sy, sz,
                 self.nxyz(),
                 '/'.join(self.get_variables()),
                 data_type, self.nvariables)

        # Replace np.nan by missing_value
        if missing_value is not None:
            np.putmask(data, np.isnan(data), missing_value)

        # Open the file in write binary mode
        with open(output_name, 'wb') as ff:
            ff.write(shead.encode())
            # Write variable values
            np.savetxt(ff, data.reshape(1, -1).T, delimiter=' ', fmt=fmt)

        # Replace missing_value by np.nan (restore)
        if missing_value is not None:
            np.putmask(data, data == missing_value, np.nan)

    # ------ Pgm ------
    @staticmethod
    def fromPgm(file_name: str, missing_value=None, var_name=['pgmValue']):
        """
        Image static method. Used as an initializer.
        Builds the container from a .pgm file.

        Parameters
        ----------
        'file_name' : string
            name of the file

        'missing_value' : float|None
            value that will be replaced by nan

        Returns
        ----------
        A new Image object
        """

        # Check if the file exists
        if not os.path.isfile(file_name):
            raise Exception("Error: invalid filename ({})".format(filename))

        # Open the file in read mode
        with open(file_name, 'r') as ff:
            # Read 1st line
            line = ff.readline()
            if line[:2] != 'P2':
                raise Exception("Error: invalid format (first line)")

            # Read 2nd line
            line = ff.readline()
            while line[0] == '#':
                # Read next line
                line = ff.readline()

            # Set dimension
            nx, ny = [int(x) for x in line.split()]

            # Read next line
            line = ff.readline()
            if line[:3] != '255':
                print("Error: invalid format (number of colors / max val)")
                return

            # Read the rest of the file
            vv = [x.split() for x in ff.readlines()]

        # Set variable array
        val_arr = np.array([int(x) for line in vv for x in line],
                           dtype=float).reshape((nx, ny, 1))
        params = {
            "origin": (0., 0., 0.),
            "spacing": (1., 1., 1.),
            "is3D": False,
            "nVariables": 1
        }

        # Replace missing_value by np.nan
        if missing_value is not None:
            np.putmask(val_arr, val_arr == missing_value, np.nan)
        return Image({"V0": val_arr}, params)

    def exportAsPgm(self, output_name: str, var_name=None,
                    missing_value=None, fmt="%.10g"):
        """
        Export the Image object data as a  pgm file.

        Parameters
        ----------

        'output_name' : string
            name of the file to be written

        'var_name' : string
            The variable to be exported. Needs to be not None if several
            variables are present in the Image class.

        'missing_value' float or None
            nan values will be replaced by missing_value before writing

        'fmt' : string
            single format for variable values, of the form:
                '%[flag]width[.precision]specifier'
        """
        if var_name is None and self.nvariables > 1:
            raise UndefVarExc("exportAsPgm")
        key = var_name if var_name is not None else self.get_variables()[0]
        data = self._data[key]
        nx, ny, nz = self.shape
        ox, oy, oz = self.orig
        sx, sy, sz = self.spacing
        # Write 1st line in string shead
        shead = "P2\n# {0} {1} {2}   {3} {4} {5}   {6} {7} {8}\n\
                 {0} {1}\n255\n".format(nx, ny, nz, sx, sy, sz, ox, oy, oz)

        # Replace np.nan by missing_value
        if missing_value is not None:
            np.putmask(data, np.isnan(data), missing_value)

        # Open the file in write binary mode
        with open(output_name, 'wb') as ff:
            ff.write(shead.encode())
            # Write variable values
            np.savetxt(ff, data.reshape(1, -1).T, delimiter=' ', fmt=fmt)

        # Replace missing_value by np.nan (restore)
        if missing_value is not None:
            np.putmask(data, data == missing_value, np.nan)

    # ------ Ppm ------
    @staticmethod
    def fromPpm(file_name: str, missing_value=None, var_name=['R', 'G', 'B']):
        """
        Image static method. Used as an initializer.
        Builds the container from a .ppm file.

        Parameters
        ----------
        'file_name' : string
            name of the file

        'missing_value' : float|None
            value that will be replaced by nan

        'var_name' : list of 3 strings
            name of the different variables of the image
            (default is ['R','G','B'])

        Returns
        ----------
        A new Image object
        """

        # Check if the file exists
        if not os.path.isfile(file_name):
            raise Exception("Error: invalid filename ({})".format(filename))

        # Open the file in read mode
        with open(file_name, 'r') as ff:
            # Read 1st line
            line = ff.readline()
            if line[:2] != 'P3':
                print("Error: invalid format (first line)")
                return

            # Read 2nd line
            line = ff.readline()
            while line[0] == '#':
                # Read next line
                line = ff.readline()

            # Set dimension
            nx, ny = [int(x) for x in line.split()]

            # Read next line
            line = ff.readline()
            if line[:3] != '255':
                raise Exception(
                    "Error: invalid format (number of colors / max val)")

            # Read the rest of the file
            vv = [x.split() for x in ff.readlines()]

        # Replace missing_value by np.nan
        if missing_value is not None:
            np.putmask(val_arr, val_arr == missing_value, np.nan)

        # Set variable array
        val_arr = np.array([float(x) for line in vv for x in line],
                           dtype=float).reshape((nx, ny, 1, 3))
        data = dict([(var_name[i], val_arr[:, :, :, i]) for i in range(3)])
        params = {"is3D": False, "nVariables": 3}
        return Image(data, params)

    def exportAsPpm(self, output_name: str, var_name=['R', 'G', 'B'],
                    missing_value=None, fmt="%.10g"):
        """
        Export the Image object data as a  pgm file.

        Parameters
        ----------
        'output_name' : string
            name of the file to be written

        'var_name' : list of 3 strings
            The names of the channels to be output.
            There should be three channels

        'missing_value' : float or None
            nan values will be replaced by missing_value before writing

        'fmt' : string
            single format for variable values, of the form:
                '%[flag]width[.precision]specifier'
        """
        # Check if the file is RGB
        if self.nvariables < 3:
            raise Exception("Ppm file uses RGB channels. Only {} variables \
                            were found. To export only one variable, please \
                            use the pgm file format \
                            instead".format(self.nvariables))

        # Write 1st line in string shead
        nx, ny, nz = self.shape
        sx, sy, sz = self.spacing
        ox, oy, oz = self.orig
        r_data, g_data, b_data = (self._data[k] for k in var_name)
        data = np.array([r_data, g_data, b_data])
        shead = "P3\n# {0} {1} {2}   {3} {4} {5}   {6} {7} {8}\n\
            {0} {1}\n255\n".format(nx, ny, nz, sx, sy, sz, ox, oy, oz)

        # Replace np.nan by missing_value
        if missing_value is not None:
            np.putmask(data, np.isnan(data), missing_value)

        # Open the file in write binary mode
        with open(output_name, 'wb') as ff:
            ff.write(shead.encode())
            # Write variable values
            np.savetxt(ff, data.reshape(3, -1).T, delimiter=' ', fmt=fmt)

        # Replace missing_value by np.nan (restore)
        if missing_value is not None:
            np.putmask(data, data == missing_value, np.nan)

    # ------ Misc Export ------
    def plot(self, name_var=None):
        """
        Calls appropriate plot method for 2D and 3D image
        """
        if self.is3D:
            self.cutplot(name_var)
        else:
            self.plot_2D(name_var)

    def plot_2D(self, name_var=None):
        """
        Displays the image using matplotlib.pyplot
        """
        import matplotlib.pyplot as plt
        if self.nvariables > 1 and name_var is None:
            raise UndefVarExc("plot")
        if self.nvariables == 1:
            data = list(self._data.values())[0]
        else:
            data = self._data[name_var]
        plt.imshow(data[:, :, 0])
        plt.colorbar()
        plt.show()

    def cutplot(self, name_var=None, cut_position=0):
        """
        Displays 3 perpendicular cuts through image using matplotlib.pyplot
        """
        import matplotlib.pyplot as plt
        if self.nvariables > 1 and name_var is None:
            raise UndefVarExc("plot")
        if self.nvariables == 1:
            data = list(self._data.values())[0]
        else:
            data = self._data[name_var]

        if self.is3D:
            plt.subplot(131)
            plt.title('0,1 section, '+'axis 2 at='+str(cut_position))
            plt.imshow(data[:, :, cut_position])
            plt.subplot(132)
            plt.title('0,2 section, '+'axis 1 at='+str(cut_position))
            plt.imshow(data[:, cut_position, :])
            plt.subplot(133)
            plt.title('1,2 section, '+'axis 0 at='+str(cut_position))
            plt.imshow(data[cut_position, :, :])
            plt.show()
        else:
            self.plot()

    def exportCuts(self,
                   output_folder="cut_output",
                   var_name: str = None,
                   axis=-1,
                   n=-1,
                   invert=False):
        """
        Export the Image object data as cuts along an axis.
        Requires the image to be a black and white one (only one channel).
        The exported cuts are saved as png files.
        If the image is two dimensionnal, performs as exportAsPng.

        Parameters
        ----------
        'var_name' : str
            The name of the variable to take cuts from.
            Needs to be provided if several variables exist.

        'output_folder' : string
            relative path to the folder file in which the cuts will be saved

        'axis' : intI
            The axis along the cuts are made.
            If set to -1, will perform cuts along all axis.
            default=-1

        'n' : int
            The number of cuts performed in the given direction.
            If set to -1, will perform every possible cuts.
            Otherwise, takes n cuts at random.
            Default=-1

        'invert' : boolean
            if set to true, will invert the colors of the image (x -> 255-x)
            default=False
        """
        from os.path import join as pj
        try:
            os.mkdir(output_folder)
        except ImportError:
            pass

        if self.nvariables > 1 and var_name is None:
            raise UndefVarExc("exportCuts")
        key = var_name if var_name is not None else list(self._data.keys())[0]
        array = self._data[key]
        if invert:
            array = -array
        iter = range(array.shape[0]) if n == -1 \
            else np.random.randint(array.shape[0], size=n)
        if axis == 1:
            for i in iter:
                img = array[:, i, :]
                self.exportAsPng(pj(output_folder, "cut_y_{}.png".format(i)))
        elif axis == 2:
            for i in iter:
                img = Image.fromArray(array[:, :, i])
                self.exportAsPng(pj(output_folder, "cut_z_{}.png".format(i)))
        elif axis == 0:
            for i in iter:
                img = Image.fromArray(array[i, :, :])
                self.exportAsPng(pj(output_folder, "cut_x_{}.png".format(i)))
        else:
            for i in iter:
                Image.fromArray(array[i, :, :]).exportAsPng(
                    pj(output_folder, "cut_x_{}.png".format(i)))
                Image.fromArray(array[:, i, :]).exportAsPng(
                    pj(output_folder, "cut_y_{}.png".format(i)))
                Image.fromArray(array[:, :, i]).exportAsPng(
                    pj(output_folder, "cut_z_{}.png".format(i)))

    # ------ Setters and getters -------

    def reset_var_names_to_default(self):
        """Sets default variable names: var_name = ('V0', 'V1',...)."""
        i = 0
        keys = list(self._data.keys())
        for old_key in keys:
            new_key = "V{:d}".format(i)
            i += 1
            self._data[new_key] = self._data.pop(old_key)

    def nxyz(self):
        return (self.shape[0] * self.shape[1] * self.shape[2])

    def nxy(self):
        return (self.shape[0] * self.shape[1])

    def nxz(self):
        return (self.shape[0] * self.shape[2])

    def nyz(self):
        return (self.shape[1] * self.shape[2])

    def xmin(self):
        return (self.orig[0])

    def ymin(self):
        return (self.orig[1])

    def zmin(self):
        return (self.orig[2])

    def xmax(self):
        return (self.orig[0] + self.spacing[0] * self.shape[0])

    def ymax(self):
        return (self.orig[1] + self.spacing[1] * self.shape[1])

    def zmax(self):
        return (self.orig[2] + self.spacing[2] * self.shape[2])

    def _dim(self, axis):
        return (self.orig[axis] + 0.5 * self.spacing[axis] +
                self.spacing[axis] * np.arange(self.shape[axis]))

    def x(self):
        """Returns 1-dimensional array of x coordinates."""
        return self._dim(0)

    def y(self):
        """Returns 1-dimensional array of y coordinates."""
        return self._dim(1)

    def z(self):
        """Returns 1-dimensional array of z coordinates."""
        return self._dim(2)

    def vmin(self, var_name: str = None):
        """
        Minimal value of a variable.

        Parameters
        ----------
        'var_name' : string
            The variable to consider. Has to be provided if there is more
            than one variable in the Image.
        """
        if var_name is None and self.nvariables > 1:
            raise UndefVarExc("vmmin")
        key = var_name if var_name is not None else self.get_variables()[0]
        return (np.nanmin(self._data[key].reshape(self.nxyz()), axis=0))

    def vmax(self, var_name: str = None):
        """
        Maximal value of a variable.

        Parameters
        ----------
        'var_name' : string
            The variable to consider. Has to be provided if there is more
            than one variable in the Image.
        """
        if var_name is None and self.nvariables > 1:
            raise UndefVarExc("vmax")
        key = var_name if var_name is not None else self.get_variables()[0]
        return (np.nanmax(self._data[key].reshape(self.nxyz()), axis=0))

    def get_variables(self) -> list:
        return list(self._data.keys())

    def add_variable(self, var_name: str = None, value=None) -> None:
        """
        Appends one variable to the image

        Parameters
        ----------
        'var_name' : string | None
            Name of the appended variable (set by default if None)

        'v' :   int/float | tuple/list/ndarray
            value(s) of the newvariable:
            if type is int/float: constant variable
            if tuple/list/ndarray: must contain nx*ny*nz values,
                        which are appended in the image (after reshape
                        if needed)

        """

        # handle value
        if value is None:
            ar = np.zeros(self.shape)
        elif isinstance(value, int) or isinstance(value, float):
            ar = np.array([value]*self.nxyz()).reshape(self.shape)
        else:
            ar = np.asarray(value, dtype=float)

        # handle var_name
        if var_name is None:
            var_name = "V"+str(len(self._data.keys()))
        self._data[var_name] = ar
        self.nvariables = len(self._data.keys())

    def set_variable(self, var_name: str, value):
        """
        Sets one variable (of given name):

        Parameters
        ----------
        var_name : string
            variable name. If the name is not present in the image,
            nothing happens

        v : int/float | tuple/list/ndarray
            value(s) of the new variable:
            if type is int/float: constant variable
            if tuple/list/ndarray: must contain nx*ny*nz values,
                which are appended in the image (reshaped if needed)
        """

        if var_name not in self._data:
            return
        if value is None:
            ar = np.zeros(self.shape)
        elif isinstance(value, int) or isinstance(value, float):
            nx, ny, nz = self.shape
            ar = [value]*(nx*ny*nz)
            ar = np.array(ar).reshape(self.shape)
        else:
            ar = np.asarray(value, dtype=float)
        assert ar.shape == self.shape
        self._data[var_name] = ar

    def rename_variable(self, old_var_name: str, new_var_name: str) -> None:
        """
        Rename one variable.
        Parameters
        ----------
        'old_var_name' : string
            The name of the variable to be replaced.
            If old_var_name does not exist in the Image, nothing happens

        'new_var_name' : string
            The new name of the variable
        """
        if old_var_name not in self._data:
            return
        self._data[new_var_name] = self._data.pop(old_var_name)

    def remove_variable(self, var_name: str) -> None:
        """
        Delete one variable of the image

        Parameters
        ----------
        'var_name'  string
            Name of the variable to delete. If this variable is not present in
            the image, nothing happens
        """
        self._data.pop(var_name, None)
        self.nvariables = len(self._data.keys())

    def extract_variable(self, var_name: list, copy=True):
        """
        Creates a new Image containing only the extracted variable

        Parameters
        ----------
        'var_name': list
            List of the names of variables to be extracted

        'copy' : boolean
            If set to False, will delete the variables of var_name
            from the original imageself.

        Returns
        -------
        A new Image object
        """
        assert len(var_name) > 0
        new_image = Image.empty(self.shape)
        for v in var_name:
            new_image._data[v] = np.copy(self._data[v])
            if not copy:
                self.remove_variable(v)
        return new_image

    # ------ Transformation methods ------

    def apply_fun(self, var_name: str = None, fun=None):
        """
        Transformation method. Applies a function to every element of the
        data container.

        Parameters
        ----------
        'var_name' : str
            The name of the variable to apply the function to.
            Needs to be provided if several variables exist.

        'fun' : a python function returning an number
            the function to be called
        """
        if fun is None:
            return
        if var_name is None and self.nvariables > 1:
            raise UndefVarExc("apply_fun")
        key = var_name if var_name is not None else list(self._data.keys())[0]
        for x in np.nditer(self._data[key], op_flags=['readwrite']):
            x[...] = fun(x)

    def saturate(self, var_name: str = None, t: int = 5):
        """
        Transformation method. Applies a saturation of height t on the image,
        that is to say : sends elements with values<t to 255 and does not
        change other values.

        This is usefull for the .vox format, where 0 values are being rendered
        as transparent.

        Parameters
        ----------
        'var_name' : str
            The name of the variable to saturate.
            Needs to be provided if several variables exist.

        't' : int
            the height of the saturation.
            default = 5
        """
        def f(x): return x if x > t else 0
        self.apply_fun(var_name, f)

    def threshold(self, thresholds=[127], values=None, var_name: str = None):
        """
        Returns a categorized image according to thresholds specified.

        Parameters
        ----------
        'var_name' : str
            The name of the variable to threshold.
            Needs to be provided if several variables exist.

        'thresholds' : ndarray | list
            must be non-empty and all image values must lie between
            first and last element of threshold

        'values' : ndarray | list
            a list of size len(thresholds)+1. If this argument is not None,
            the categories will take the given. Otherwise, the categores will
            have their mean value as a category value.
        """
        if var_name is None and self.nvariables > 1:
            raise UndefVarExc("threshold")
        key = var_name if var_name is not None else list(self._data.keys())[0]
        ar = self._data[key]

        # Check if thresholds input is correct
        thresholds = sorted(thresholds)

        replaced = np.zeros(ar.shape)
        for i in range(len(thresholds)):
            i_th_cat = (ar < thresholds[i]) & (replaced == 0)
            replaced[i_th_cat] = 1
            if values is None:
                ar[i_th_cat] = np.mean(ar[i_th_cat])
            else:
                ar[i_th_cat] = values[i]
        final_cat = replaced == 0
        if values is None:
            ar[final_cat] = np.mean(ar[final_cat])
        else:
            ar[final_cat] = values[-1]

    def categorize(self,
                   nb_categories=2,
                   var_name: str = None,
                   initial_clusters=None,
                   norm="l1",
                   max_iter=10):
        """
        Transformation method. Applies a clustering algorithm to categorize the
        image. k clusters will be created, with the color value of their
        barycenter.

        Parameters
        ----------
        'var_name' : str
            The name of the variable to categorize.
            Needs to be provided if several variables exist.

        'nb_categories' : int
            The number of categories

        'initial_clusters' : list
            Values of the initial centroids for the cluster. If not provided,
            those values will be taken at random

        'norm' : str
            The distance to be used. Supported distance are l1 and l2.
            Only relevant when dealing with colored (multi channel) images.

        'max_iter' : int
            Maximal number of cluster updates allowed
        """
        if var_name is None and self.nvariables > 1:
            raise UndefVarExc("categorize")
        key = var_name if var_name is not None else list(self._data.keys())[0]
        ar = self._data[key]

        # Initialize centroids
        vmin, vmax = np.amin(ar), np.amax(ar)
        if initial_clusters is None:
            centroids = [np.random.rand()*(vmax-vmin) +
                         vmin for i in range(nb_categories)]
        else:
            centroids = initial_clusters
            np.clip(centroids, vmin, vmax)

        # Initialize categories : assign value to closest centroid
        categories, counts = np.unique(ar, return_counts=True)
        total = np.sum(counts)
        mapsto = np.zeros(categories.shape).astype(np.int16)
        for ind, val in np.ndenumerate(categories):
            mapsto[ind] = np.argmin([abs(x-val) for x in centroids])

        # Iteration while something is changing
        has_updated = True
        n = 0
        while has_updated and n < max_iter:
            has_updated = False
            n += 1
            # Update centroids positions
            for i in range(nb_categories):
                centroids[i] = np.mean(
                    [categories[pos]
                     for pos in np.ndindex(categories.shape)
                     if mapsto[pos] == i])
            # Update every value according to new centroids
            for ind, val in np.ndenumerate(categories):
                new_ind = np.argmin([abs(x-val) for x in centroids])
                if new_ind != mapsto[ind]:
                    mapsto[ind] = new_ind
                    has_updated = True

        # assign value of centroids to majority of their cluster
        for i in range(nb_categories):
            i_th_category = mapsto == i
            majority_i = np.argmax(counts[i_th_category])
            centroids[i] = categories[i_th_category][majority_i]

        # rewrite self._data with new categorical data
        mapsto_dict = dict()
        for ind, cat in np.ndenumerate(categories):
            mapsto_dict[cat] = centroids[mapsto[ind]]
        for pos in np.ndindex(ar.shape):
            ar[pos] = mapsto_dict[ar[pos]]

    def normalize(self, var_names=[]):
        """
        Transformation method. Applies a linear transformation
        to get all data in range [-1,1]

        Parameters
        ----------
        'var_names' : list of string
            The names of the variables to normalize.
            If set to empty list (default value), all variables will be
            normalized
        """
        keys = self._data.keys() if not var_names else var_names
        for key in keys:
            self._data[key] -= np.amin(self._data[key])
            m = np.amax(self._data[key])
            if abs(m) > 1e-8:
                self._data[key] = self._data[key] / m
            self._data[key] = 2*self._data[key] - 1

    def unnormalize(self, var_name: list = None, output_type=np.uint8):
        """
        Transformation method. Applies a linear transformation
        to get all data in range [0,255]

        Parameters
        ----------
        'output_type' : np.dtype
            The type the data will be casted to.
            default = np.uint8 (integers in range [0;255])

        'var_names' : list of string
            The names of the variables to normalize.
            If set to empty list (default value), all variables will be
            normalized
        """
        self.normalize()
        keys = self._data.keys() if var_name is None else var_name
        for key in keys:
            self._data[key] = (self._data[key]+1)*127.5
            self._data[key] = self._data[key].astype(output_type)

    def flipx(self):
        """ Flips variable values according to x direction. """
        for k in self._data:
            self._data[k] = self._data[k][::-1, :, :]

    def flipy(self):
        """ Flips variable values according to x direction. """
        for k in self._data:
            self._data[k] = self._data[k][:, ::-1, :]

    def flipz(self):
        """ Flips variable values according to x direction. """
        if self.is3D:
            for k in self._data:
                self._data[k] = self._data[k][:, :, ::-1]

    def _perm(self, i, j):
        for k in self._data:
            self._data[k] = np.swapaxes(self._data[k], i, j)
        new_order = [0, 1, 2]
        new_order[i], new_order[j] = new_order[j], new_order[i]
        self.orig = tuple(self.orig[i] for i in new_order)
        self.spacing = tuple(self.spacing[i] for i in new_order)
        self.shape = tuple(self.shape[i] for i in new_order)

    def permxy(self):
        """Permutes x and y directions."""
        self._perm(0, 1)

    def permxz(self):
        """Permutes x and z directions."""
        self._perm(0, 2)

    def permyz(self):
        """Permutes y and z directions."""
        self._perm(1, 2)

    def get_sample(self, output_dim,
                   var_name: list = None,
                   normalize: bool = False):
        """
        Extract a random submatrix of a given size from the data container.

        Parameters
        ----------
        'output_dim' : tuple
            The size of the sample. All coordinates should lay between 0 and
            the corresponding coordinate of self._data

        'normalize' : boolean
            if set to true, apply the normalize method to the output sample to
            get values in [-1;1]

        'var_name' : list
            the name of the variables in which the sample is taken.
            Needs to be specified if nvariables>1

        Returns
        ----------
        A new Image object of size (output_dim * len(var_name))
        """
        if var_name is None:
            if self.nvariables > 1:
                raise UndefVarExc("get_sample")
            var_name = list(self._data.keys())
        xd, yd, zd = output_dim
        sample = {}
        if self.is3D:
            xs, ys, zs = self.shape
            choice_x = np.random.randint(xs-xd)
            choice_y = np.random.randint(ys-yd)
            choice_z = np.random.randint(zs-zd)
            for name in var_name:
                sample[name] = self._data[name][choice_x:choice_x+xd,
                                                choice_y:choice_y+yd,
                                                choice_z:choice_z+zd]
        else:
            xs, ys = self.shape[0], self.shape[1]
            choice_x = np.random.randint(xs-xd)
            choice_y = np.random.randint(ys-yd)
            for name in var_name:
                sample[name] = self._data[name][choice_x:choice_x+xd,
                                                choice_y:choice_y+yd,
                                                0:1]
        params = dict([('is3D', self.is3D)])
        sample = Image(sample, params)
        if normalize:
            sample.normalize()
        return sample

    @staticmethod
    def tile_images(image_stack, mode):
        """
        Given a list of Images, reshapes them into a tiling for display.

        Parameters
        ----------
        'image_stack' : list of Images
            the images that will be concatenated

        'mode' : string | tuple
            The mode of tiling. Three predefined modes are available :
              horizontal -> 'h' option
              vertical   -> 'v' option
              square     -> 's' option
            You can also provide a tuple (nb_lines,nb_columns) for any
            rectangular tiling. If your number of images is not
            nb_lines*nb_columns, the function will complete with black images

        Returns
        -------
        A new Image instance
        """
        image_stack = [img.asArray() for img in image_stack]
        shape = None
        N = len(image_stack)
        if mode in ["horizontal", 'h']:
            shape = (1, N)
        elif mode in ["vertical", 'v']:
            shape = (N, 1)
        elif mode in ["square", 's']:
            n = int(sqrt(N))
            m = n if n*n == N else n+1
            shape = (m, m)
        else:
            assert (len(mode) == 2)
            x, y = mode
            if x*y < N:
                while x*y < N:
                    y += 1
            elif x*y > N:
                while x*y > N:
                    y -= 1
            shape = (x, y)
        imgshape = image_stack[0].shape
        blacks = np.zeros(imgshape)
        for i in range(shape[0]*shape[1]-N):
            image_stack.append(blacks)
        rows = []
        for i in range(shape[0]):
            rows.append(np.concatenate(
                image_stack[i*shape[1]:(i+1)*shape[1]], axis=1))
        return Image.fromArray(np.concatenate(rows, axis=0))


def labelize(image):
    """
    Label the data to get integers from 0 to the number of facies

    Parameters
    ----------
    image : ndarray | Image
        non-empty numpy array or Image class object

    Returns
    -------
    ndarray
        array of the same shape of image containing the categories
    """
    data = image.asArray() if isinstance(image, Image) else image
    output = np.zeros(data.shape).astype(np.int32)
    facies = np.unique(data)
    labels = dict([(val, ind[0]) for ind, val in np.ndenumerate(facies)])
    for pos in np.ndindex(data.shape):
        output[pos] = labels[data[pos]]
    if output.shape[-1] == 1:
        output = output.reshape(output.shape[:-1])
    return output
