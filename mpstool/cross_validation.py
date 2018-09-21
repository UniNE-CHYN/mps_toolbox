#!/usr/bin/env python3

import numpy as np
import subprocess

class ConditioningData():
    """
    Stores conditioning data
    """
    def __init__(self, pixels, values):
        if len(pixels) == len(values):
            self.pixels = pixels
            self.values = values
            self.number_of_points = len(pixels)
        else:
            raise ValueError("Pixels size and values size must match")
    def __eq__(self, other):
        if isinstance(other, ConditioningData):
            return (self.pixels == other.pixels and self.values == other.values and self.number_of_points == other.number_of_points)
        else:
           return False

        
def sample_random_conditioning_data(image, number_of_points):
    if number_of_points <= image.size:
        # pick random indexes
        indexes = np.random.permutation(image.size)[:number_of_points]

        # get values corresponding to these indexes
        nx, ny = image.shape
        pixels = [ (index%nx, int((index-(index%nx))/nx))  for index in indexes]
        values = [image[pixel[0], pixel[1]] for pixel in pixels]
        return ConditioningData(pixels, values)
    else:
        raise ValueError("Number of points must not exceed image size")
    
def save_to_gslib(filename, conditioning_data):
    with open(filename, 'w') as data_file:
        data_file.write(str(conditioning_data.number_of_points)+'\n')
        data_file.write(str(3)+'\n')
        data_file.write('x'+'\n')
        data_file.write('y'+'\n')
        data_file.write('value'+'\n')
        for (pixel, value) in zip(conditioning_data.pixels, conditioning_data.values):
            data_file.write(str(pixel[0]) + " " + str(pixel[1]) + " " + str(value)+'\n')

def read_from_gslib(filename):
    pixels = []
    values = []
    with open(filename) as data_file:
        # First line: Nx Ny Nz [Sx Sy Sz [Ox Oy Oz]]
        data_file.readline()

        # Second line: number of variables
        nvar = int(data_file.readline())

        # Names of variables 1..nvar
        for i in range(nvar):
            data_file.readline()

        # Read the data in 3 columns
        for line in data_file:
            line = line.split()
            pixels.append( (int(line[0]), int(line[1])) )
            values.append(float(line[2]))
    return ConditioningData(pixels, values)

def plot(conditioning_data):
    import matplotlib.pyplot as plt

    X = [pixel[0] for pixel in conditioning_data.pixels]
    Y = [pixel[1] for pixel in conditioning_data.pixels]
    plt.scatter(X, Y, c=conditioning_data.values)
    plt.show()

def cross_validation(ti_filename, data_filename, n):
    data = read_from_gslib(data_filename)

    temp_filename = 'temp.gslib'
    print(data.number_of_points)
    # order randomly data points
    permutation = np.asarray(np.random.permutation(data.number_of_points), dtype=int)
    print(type(permutation))
    conditioning_data = ConditioningData([data.pixels[i] for i in permutation], [data.values[i] for i in permutation])

    for i in range(1, conditioning_data.number_of_points-1):
        available_data = ConditioningData(conditioning_data.pixels[:i], conditioning_data.values[:i])
        save_to_gslib(temp_filename, available_data)
        simulation_process = subprocess.Popen([name] + arguments, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, err = simulation_process.communicate()

