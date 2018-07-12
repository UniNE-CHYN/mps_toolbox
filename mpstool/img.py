import numpy as np
import os
from math import sqrt

class Image:
    """
    A data container.
    The underlying data structure is a numpy ndarray.
    Implements transformations on images, as well as type conversion
    from the following data types :

    png
    ---------
    The classical image extension


    gslib
    ----------
    The .gslib format is a text format with the following structure :

     nx  ny  nz  x_orig  y_orig  z_orig x_padding y_padding z_padding
     number_of_variables (should be 1 for our use)
     name_of_variable_1
     ...
     name_of_variable_n

     then, on each line, the value of a coordinate (x,y,z), in the order of the
     nested loop :
     for x = 1 to nx
         for y = 1 to ny:
             for z = 1 to nz:


    vox
    ----------
    A binary file used in MagicaVoxel and in other voxel editors.
    See : https://ephtracy.github.io/index.html?page=mv_main
    """

    def __init__(self, data, params):
        """
        Initialisation method.
        It is not meant to be called.
        Instead, use the fromGslib, fromArray, fromPng and fromVox methods
        """
        self._data = data
        self.is3D = params["is3D"]
        self.isColored = params["isColored"]
        self.orig = (0,0,0)
        if "origin" in params:
            self.orig = params["origin"]
        self.padding = (1,1,1)
        if "padding" in params:
            self.padding = params["padding"]
        if self.is3D:
            self.shape = self._data.shape
        else:
            self.shape = self._data.shape[:3]
            if len(self._data.shape)==2:
                self.shape = self.shape + (1,)
                self._data = self._data.reshape(self.shape)
            elif len(self._data.shape)==3 and self._data.shape[2]==4:
                self._data = self._data[:,:,:3] # get rid of transparency

    def __eq__(self,other):
        return np.alltrue(self._data == other._data)

    ## ------- Import methods
    @staticmethod
    def fromGslib(file_name, normalize=False):
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
        with open(file_name, 'r') as f :
            params = dict()
            metadata = f.readline().strip().split()
            xdim = int(metadata[0])
            ydim = int(metadata[1])
            zdim = int(metadata[2])
            params["is3D"]= (zdim>1)
            params["isColored"]=False
            if len(metadata)>3:
                xstep = float(metadata[3])
                ystep = float(metadata[4])
                zstep = float(metadata[5])
                params["padding"]=(xstep,ystep,zstep)
                xorig = float(metadata[6])
                yorig = float(metadata[7])
                zorig = float(metadata[8])
                params["origin"]=(xorig,yorig,zorig)
            nb_var = int(f.readline().strip())
            _ = f.readline()
            ar = np.zeros((xdim,ydim,zdim))
            for iz in range(zdim):
                for iy in range(ydim):
                    for ix in range(xdim):
                        ar[ix,iy,iz] = float(f.readline().strip())
        img = Image(ar,params)
        if normalize:
            img.normalize()
        return img

    @staticmethod
    def fromArray(ar):
        """
        Image staticmethod. Used as an initializer.
        Builds the container from a numpy array

        Parameters
        ----------
        'ar' : ndarray | list of ndarray
            The numpy array around which the Image object is built
            A list of 2D ndarray can be given in order to build a 3D image

        Returns
        ----------
        A new Image object
        """
        if isinstance(ar,list):
            ar = np.array(ar)
            if len(ar.shape)>3:
                ar = ar[:,:,:,0]
        shape = ar.shape
        params = dict()
        if len(shape)<3 or shape[2]==1:
            params["isColored"] = False
            params["is3D"] = False
        elif shape[2]==3:
            params["isColored"] = True
            params["is3D"] = False
        else:
            params["isColored"] = False
            params["is3D"] = True
        return Image(ar,params)

    @staticmethod
    def fromTxt(file_name,shape):
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
        return Image.fromArray(array)

    @staticmethod
    def fromPng(file_name, normalize=False):
        """
        Image staticmethod. Used as an initializer.
        Builds the container from a .png file.
        Makes calls to the Pillow library

        Parameters
        ----------
        'file_name' : string
            relative path to the png file

        'normalize' : boolean
            if set to true, the values will be stretched to fit in [-1;1]

        Returns
        ----------
        A new Image object
        """
        try:
            from PIL import Image as PIL_Img
        except:
            print("Cannot read from png. Is the pillow library installed ?\n\
                   To install it, run `pip install pillow`")
            return
        data = PIL_Img.open(file_name)
        data = np.array(data).astype(np.float32)
        params = dict()
        params["is3D"]=False
        params["isColored"]= len(data.shape)==3 and data.shape[2]==3
        img = Image(data, params)
        if normalize:
            img.normalize()
        return img

    @staticmethod
    def fromVox(file_name):
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
        except:
            print("py-vox-io is not installed. Cannot import a vox file.\n\
                  Please install py-vox-io with `pip install py-vox-io`")
            return
        from pyvox.writer import VoxWriter
        data = VoxParser(file_name).parse().to_dense()
        params = dict([("is3D",data.shape[2]>1),("isColored",False)])
        return Image(data, params)


    ## ------- Export methods
    def asArray(self):
        """
        Return the raw data as a numpy array
        """
        return self._data

    def exportAsTxt(self,output_name, verbose=False):
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
        if self.is3D:
            print("ERROR : Export as a txt file requires the data \
                   to be 2 dimensionnal.")
            return
        output = self._data.reshape(self.shape[:2])
        np.savetxt(output_name, output)
        if verbose:
            print("Generated txt file as {}".format(output_name))

    def exportAsPng(self, output_name, verbose=False):
        """
        Export the Image object data as a png file.
        Requires the data to be two dimensionnal.

        Parameters
        ----------
        'output_name' : string
            relative path to the png file to be output
        'verbose' : boolean
            enables verbose mode. Set to False by default
        """
        try:
            from PIL import Image as PIL_Img
        except Exception as e:
            print("ERROR : Cannot export as a png. Received the following error :\n{}\n\
                   Is the pillow library installed ?\n\
                   To install it, run `pip install pillow`".format(e))
            return
        if self.is3D:
            print("ERROR : Export as a png file requires the data \
                   to be 2 dimensionnal.")
            return
        self.unnormalize()
        if self.isColored:
            output = PIL_Img.fromarray(self.asArray())
            output.save(output_name)
        else:
            output = PIL_Img.fromarray(np.squeeze(self.asArray()))
            output.save(output_name, mode="L") # L for greyscale
        if verbose:
            print("Generated image as {}".format(output_name))

    def exportAsGslib(self, output_name, verbose=False):
        """
        Export the Image object data as a gslib file.
        Requires the image to be a black and white one (only one channel)

        Parameters
        ----------
        'output_name' : string
            relative path to the gslib file to be output
        'verbose' : boolean
            enables verbose mode. Set to False by default
        """
        if self.isColored:
            print("[ERROR] Image class : unable to convert colored images into gslib. Aborting")
            exit(0)
        with open(output_name, 'w') as f :
            xdim = self.shape[0]
            ydim = self.shape[1]
            if self.is3D:
                zdim=self.shape[2]
            else:
                zdim=1
            f.write("{} {} {} 1 1 1 0 0 0\n1\ncode\n".format(xdim,ydim,zdim))
            for iz in range(zdim):
                for iy in range(ydim):
                    for ix in range(xdim):
                        f.write(str(self._data[ix,iy,iz])+"\n")
        if verbose:
            print("Generated .gslib file as {}".format(output_name))

    def exportAsVox(self, output_name, verbose=False):
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
        except:
            print("py-vox-io is not installed. Cannot export as vox file.\n\
                  Please install py-vox-io with `pip install py-vox-io`")
            return
        self.unnormalize()
        # Crop to 255, otherwise conversion fails because input is too big
        # (Dimension has to fit in uint8 data type)
        a = self._data[:255,:255,:255]
        if ((np.array(self.shape)>255).any()):
            print("[WARNING] the image shape {} is to big to be converted into vox.\n\
                  It will be cropped at 255.")
        vox = Vox.from_dense(a)
        VoxWriter(output_name, vox).write()
        if verbose:
            print("Generated .vox file as {}".format(output_name))


    def exportCuts(self, output_folder="cut_output", axis=-1, n=-1, invert=False):
        """
        Export the Image object data as cuts along an axis.
        Requires the image to be a black and white one (only one channel).
        The exported cuts are saved as png files.
        If the image is two dimensionnal, performs as exportAsPng.

        Parameters
        ----------
        'output_folder' : string
            relative path to the folder file in which the cuts will be saved

        'axis' : intI
            The axis along the cuts are made. If set to -1, will perform cuts along
            all axis.
            default=-1

        'n' : int
            The number of cuts performed in the given direction. If set to -1, will
            perform every possible cuts. Otherwise, takes n cuts at random.
            default=-1

        'invert' : boolean
            if set to true, will invert the colors of the image (x -> 255-x)
            default=False
        """
        from os.path import join as pj
        try:
            os.mkdir(output_folder)
        except:
            pass
        array = self.asArray()
        if invert: array = -array
        iter = range(array.shape[0]) if n==-1 else np.random.randint(array.shape[0],size=n)
        if axis==1:
            for i in iter:
                img = array[:,i,:]
                save_image(img, pj(output_folder,"cut_y_{}.png".format(i)))
        elif axis==2:
            for i in iter:
                img = array[:,:,i]
                save_image(img, pj(output_folder,"cut_z_{}.png".format(i)))
        elif axis==0:
            for i in iter:
                img = array[i,:,:]
                save_image(img, pj(output_folder,"cut_x_{}.png".format(i)))
        else:
            for i in iter:
                save_image(array[i,:,:], pj(output_folder,"cut_x_{}.png".format(i)))
                save_image(array[:,i,:], pj(output_folder,"cut_y_{}.png".format(i)))
                save_image(array[:,:,i], pj(output_folder,"cut_z_{}.png".format(i)))


    ## -------- Transformation methods
    def apply_fun(self,fun):
        """
        Transformation method. Applies a function to every element of the data container.
        Parameters
        ----------
        'fun' : a python function returning an number
            the function to be called
        """
        for x in np.nditer(self._data, op_flags=['readwrite']):
            x[...]=fun(x)

    def threshold(self,t=127):
        """
        Transformation method. Applies a threshold of height t on the image, that is to say :
        sends elements with values<t to 0 and other values to 255
        Parameters
        ----------
        't' : int
            the height of the threshold.
            default = 127
        """
        assert t>=0 and t<256
        f = lambda x : 0 if x<=t else 255
        self.apply_fun(f)

    def saturate_white(self,t=250):
        """
        Transformation method. Applies a saturation of height t on the image, that is to say :
        sends elements with values>t to 255 and does not change other values

        Parameters
        ----------
        't' : int
            the height of the saturation.
            default = 250
        """
        f = lambda x : x if x<t else 255
        self.apply_fun(f)

    def normalize(self):
        """
        Transformation method. Applies a linear transformation
        to get all data in range [-1,1]
        """
        self._data -= np.amin(self._data)
        m = np.amax(self._data)
        if abs(m)>1e-8:
            self._data = self._data / m
        self._data = 2*self._data - 1

    def unnormalize(self, output_type=np.uint8):
        """
        Transformation method. Applies a linear transformation
        to get all data in range [0,255]

        Parameters
        ----------
        'output_type' : np.dtype
            The type the data will be casted to.
            default = np.uint8 (integers in range [0;255])
        """
        self.normalize()
        self._data = (self._data+1)*127.5
        self._data = self._data.astype(output_type)

    def get_sample(self, output_dim, normalize=False):
        """
        Extract a random submatrix of a given size from the data container.

        Parameters
        ----------
        'output_dim' : tuple
            The size of the sample. All coordinates should lay between 0 and the
            corresponding coordinate of self._data

        'normalize' : boolean
            if set to true, apply the normalize method to the output sample to
            get values in [-1;1

        Returns
        ----------
        A new Image object of size 'output_dim'
        """
        xd,yd,zd = output_dim
        is_colored = not output_dim[2]==1
        if self.is3D:
            xs,ys,zs = self.shape
            choice_x = np.random.randint(xs-xd)
            choice_y = np.random.randint(ys-yd)
            choice_z = np.random.randint(zs-zd)
            sample = self._data[choice_x:choice_x+xd, choice_y:choice_y+yd,choice_z:choice_z+zd]
        else:
            xs,ys = self.shape[0],self.shape[1]
            choice_x = np.random.randint(xs-xd)
            choice_y = np.random.randint(ys-yd)
            if is_colored:
                sample = self._data[choice_x:choice_x+xd, choice_y:choice_y+yd]
            else:
                sample = self._data[choice_x:choice_x+xd, choice_y:choice_y+yd,0:1]
        params = dict([ ('is3D',self.is3D),('isColored',is_colored)])
        sample = Image(sample,params)
        if normalize:
            sample.normalize()
        return sample

    @staticmethod
    def tile_images(image_stack, mode='h'):
        """
        Given a list of Images, reshapes them into a tiling for display.

        Parameters
        ----------
        'image_stack' : list of Images
            the images that will be concatenated

        'mode' : string
            The mode of tiling. Three modes are available :
              horizontal -> 'h' option
              vertical   -> 'v' option
              square     -> 's' option
            default mode is horizontal
        """
        image_stack = [img.asArray() for img in image_stack]
        if mode in ["horizontal", 'h']:
            return Image.fromArray(np.concatenate(image_stack, axis=1))
        elif mode in ["vertical", 'v']:
            return Image.fromArray(np.concatenate(image_stack, axis=0))
        elif mode in ["square", 's']:
            N = len(image_stack)
            n = int(sqrt(N))
            m = n if n*n==N else n+1
            imgshape = image_stack[0].shape
            blacks = np.zeros(imgshape)
            for i in range(N-n*m):
                image_stack.append(blacks)
            image_stack = np.array(image_stack)
            image_stack = image_stack.reshape((n,m)+imgshape)
            image_stack = np.concatenate(image_stack, axis=1)
            image_stack = np.concatenate(image_stack, axis=1)
            print(image_stack.shape)
            return Image.fromArray(image_stack)
        else:
            raise Exception("tiling mode should be 'horizontal' ('h'),  'vertical' ('v') or 'square' ('s')")

## ------------------ Conversion functions -------------------------------------

def gslib_to_png(gslib_file, output_name):
    try:
        from PIL import Image as PIL_Img
    except:
        print("ERROR : is the pillow library installed ?")
        return
    with open(gslib_file, 'r') as f :
        metadata = f.readline().strip().split()
        xdim = int(metadata[0])
        ydim = int(metadata[1])
        zdim = int(metadata[2])
        nb_var = int(f.readline().strip())
        var_name = f.readline().strip()
        for iz in range(zdim):
            V = np.zeros((xdim,ydim))
            for iy in range(ydim):
                for ix in range(xdim):
                    V[ix,iy]=float(f.readline().strip())
            V *= 255/np.amax(V)
            V = np.squeeze(np.round(V).astype(np.uint8))
            output = PIL_Img.fromarray(V, mode='L')
            output.save('{}_{}.png'.format(output_name,iz))

def png_to_gslib(png_file, output_name):
    img = Image.fromPng(png_file)
    img.exportAsGslib(output_name)

def gslib_to_vox(in_file, out_file, verbose=False):
    img = Image.fromGslib(in_file)
    img.exportAsVox(out_file, verbose)

def vox_to_gslib(in_file,out_file):
    img = Image.fromVox(in_file)
    img.exportAsGslib(out_file)

def gslib_to_cuts(gslib_file, output_folder="cut_output", axis=-1, n=-1, invert=False):
    img = Image.fromGslib(gslib_file)
    img.exportCuts(output_folder=output_folder, axis=axis, n=n, invert=invert)
