#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import cc

def subimage(image, nx, ny):
    """
    Returns a random subimage of an image of size nx x ny
    """
    x = np.random.randint(image.shape[0]-nx)
    y = np.random.randint(image.shape[1]-ny)
    return image[x:x+nx,y:y+ny]



def loadtxt(filename, nx, ny):
    array = np.loadtxt(filename)
    return Image(array, nx, ny)

class Image:
    def __init__(self, array, nx, ny):
        if len(array) != nx*ny:
            raise ValueError("Wrong size of array", len(array), nx*ny)
        self.image = array
        self.size = len(array)
        self.nx = nx
        self.ny = ny

    def generate_bins(self, nb_bins):
        return np.linspace(min(self.image), max(self.image), nb_bins+1)

    def generate_lag_classes(self, nb_lag_classes, max_lag):
        return np.linspace(0, max_lag, nb_lag_classes+1) 

    def reshape_2d(self):
        return np.reshape(self.image, (self.nx,self.ny))

    def discrete_x_variogram(self):
        image_2d = self.reshape_2d()
        volume = np.zeros(self.nx-1)
        squared_difference = np.zeros(self.nx-1)
        for x in np.arange(1,self.nx):
            squared_difference[x-1] = np.sum((image_2d[x:,:]-image_2d[:-x,:])**2)
            volume[x-1] = self.ny * (self.nx - x)
        distance = np.arange(1,self.nx)
        return distance, volume, squared_difference

    def variance(self):
        return np.var(self.image)

    def plot(self):
        image = self.reshape_2d()

        fig, axes = plt.subplots(nrows=1, ncols=1)
        cax = axes.imshow(image)
        fig.colorbar(cax)
        plt.show()

    def categories_list(self):
        categories_list = []
        categories_list.append(self.image[0])
        for pixel in self.image:
            if pixel in categories_list:
                pass
            else:
                categories_list.append(pixel)
        return np.array(categories_list)

    def percent_of_categories(self, categories):
        count_list = np.zeros_like(categories)
        total_count = 0
        for pixel in self.image:
            total_count +=1
            for i in np.arange(categories.shape[0]):
                if pixel == categories[i]:
                    count_list[i] += 1
        return count_list/total_count

    def indicator_variable(self, category):
        indicator = (self.image == category)
        return Image(indicator.astype(int), self.nx, self.ny)

    def components_array(self):
        cc_generator = cc.connected_components(self.reshape_2d())
        cc_generator.fill_label_array()
        return cc_generator.get_label_array()

    def discrete_x_connectivity(self, category):
        nx = self.nx
        categorical_2d = self.reshape_2d()
        components_2d = self.components_array()
        category_2d = np.zeros_like(categorical_2d)+category
        mask_2d = categorical_2d == category_2d
        same_category_count = np.zeros(nx-1)
        same_component_count = np.zeros(nx-1)
        for x in np.arange(1,nx):
            same_category_count[x-1] = np.sum(np.logical_and(categorical_2d[x:,:]==categorical_2d[:-x,:], mask_2d[x:,:]))
            same_component_count[x-1] = np.sum(np.logical_and(components_2d[x:,:]==components_2d[:-x,:],mask_2d[x:,:]))
        connectivity = np.divide(same_component_count, same_category_count, out=np.zeros_like(same_component_count), where=same_category_count!=0)
        distance = np.arange(1,nx)
        return distance, connectivity



def connectivity_function(image, lag_classes, category):
    distance, connectivity = image.discrete_x_connectivity(category)
    connectivity_in_classes = np.zeros(lag_classes.shape[0]-1)
    for i in np.arange(lag_classes.shape[0]-1):
        begin_range = lag_classes[i]
        end_range = lag_classes[i+1]
        in_range = np.logical_and(begin_range <= distance, distance < end_range)
        connectivity_in_classes[i] = np.mean(connectivity[in_range])
    return ConnectivityFunction(lag_classes, connectivity_in_classes)
    
class ConnectivityFunction:
    def __init__(self, lag_classes, values):
        self.lag_classes = lag_classes
        self.values = values
    def plot(self):
        centers = (self.lag_classes[1:] - self.lag_classes[0:-1])/2 + self.lag_classes[0:-1]
        plt.scatter(centers, self.values)
        plt.show()


def histogram(image, bins_list):
    counts, bins = np.histogram(image.image, bins = bins_list, density=True)
    return Histogram(counts, bins)

class Histogram:
    def __init__(self, counts, bins):
        self.counts = counts
        self.bins = bins

    def plot(self):
        bins_centers = (self.bins[1:] - self.bins[0:-1])/2 + self.bins[0:-1]
        plt.plot(bins_centers, self.counts)
        plt.show()


def variogram(image, lag_classes):
    distance, volume, squared_difference = image.discrete_x_variogram()
    variogram_volume = np.zeros(lag_classes.shape[0]-1)
    variogram_squared_difference = np.zeros(lag_classes.shape[0]-1)

    for i in np.arange(lag_classes.shape[0]-1):
        begin_range = lag_classes[i]
        end_range = lag_classes[i+1]
        in_range = np.logical_and(begin_range <= distance, distance < end_range)
        variogram_volume[i] = np.sum(volume[in_range])
        variogram_squared_difference[i] = np.sum(squared_difference[in_range])

    return Variogram(lag_classes, np.divide(0.5*variogram_squared_difference, variogram_volume, out=np.zeros_like(variogram_volume), where=variogram_volume!=0), image.variance())

class Variogram:
    def __init__(self, lag_classes, values, variance):
        self.lag_classes = lag_classes
        self.values = values
        self.variance = variance
    def plot(self):
        centers = (self.lag_classes[1:] - self.lag_classes[0:-1])/2 + self.lag_classes[0:-1]
        plt.scatter(centers, self.values)
        plt.show()

def categorical_histogram_error(image1, image2):
    categories = image1.categories_list()
    return np.abs(image1.percent_of_categories(categories)-image2.percent_of_categories(categories))


def connectivity_error(connectivity_function1, connectivity_function2):
    return average_difference(connectivity_function1.values, connectivity_function2.values)

def histogram_error(reference_histogram, simulated_histogram):
    if not np.array_equal(reference_histogram.bins,simulated_histogram.bins):
        raise ValueError("Histogram bins must match")
    return kullback_leibler_divergence(simulated_histogram.counts, reference_histogram.counts)

def variogram_error(ref_variogram, sim_variogram):
    if not np.array_equal(ref_variogram.lag_classes, sim_variogram.lag_classes):
        raise ValueError("Variogram lag classes must match")
    h_d = ref_variogram.lag_classes[0:-1] + (ref_variogram.lag_classes[1:]-ref_variogram.lag_classes[:-1])/2
    return weighted_average_difference(ref_variogram.values, sim_variogram.values, h_d, sim_variogram.variance)


def kullback_leibler_divergence(P, Q):
    if len(P) != len(Q):
        raise ValueError("P and Q length must match")
    nonzero = np.logical_and(P!=0, Q!=0)
    return np.sum(P[nonzero]*np.log(P[nonzero]/Q[nonzero]))

def weighted_average_difference(gamma_ti, gamma_sim, h_d, var_sim):
    return (np.sum((1/h_d)*np.abs(gamma_sim-gamma_ti)/var_sim) / np.sum(1/h_d))

def average_difference(tau_ti, tau_sim):
    return np.mean(np.abs(tau_sim-tau_ti))
