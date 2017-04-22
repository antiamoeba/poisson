#!/usr/bin/env python
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import sys, os, glob
import IPython
from scipy.interpolate import RegularGridInterpolator as RGI


# Parameters and defaults
image_num = 1



def readimage(name):
    image = plt.imread(name)
    if name[-4:] == ".jpg":
        image = image.astype(float) / 255.0
    elif name[-4:] == ".png":
        image = image[:,:,:3]
    return image

def publishImage(image, title=None, output_directory=True):
    if len(image.shape) == 2:
        image = np.dstack([image for _ in range(3)])
    else:
        if title:
            if output_directory:
                imgName = output_dir + '/' + title
            else:
                imgName = title
        else:
            global image_num
            if output_directory:
                imgName = output_dir + '/image' + str(image_num) + '.png'
            else:
                imgName = 'image' + str(image_num) + '.png'
            image_num += 1
        plt.imsave(imgName, image)
    return

def displayImage(image, title=None):
    if len(image.shape) == 2:
        image = np.dstack([image for _ in range(3)])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.axis('off')
    ax.imshow(image)
    return ax




class Panorama:
    def __init__(self):
        print "\nCS 284B Final Project: Panoramas"
        return


    def cylindricalMappingIndices(self, initial_indices, focal_length, center, scaling):
        new_x = focal_length * np.tan((initial_indices[1] - center[1]) / scaling)
        new_y = focal_length * ((initial_indices[0] - center[0]) / scaling) / np.cos((initial_indices[1] - center[1]) / scaling)
        return (new_y, new_x)


    def sphericalMappingIndices(self, initial_indices, focal_length, center, scaling):
        new_x = focal_length * np.tan((initial_indices[1] - center[1]) / scaling)
        new_y = focal_length * np.tan((initial_indices[0] - center[0]) / scaling) / np.cos((initial_indices[1] - center[1]) / scaling)
        return (new_y, new_x)


    def warpImage(self, image, focal_length, mapping, scaling=None):
        if scaling is None:
            scaling = focal_length
        result = np.zeros(image.shape)
        center = np.array(image.shape[:2]) / 2

        # Interpolation functions
        x = np.array(range(image.shape[0])) - center[0]
        y = np.array(range(image.shape[1])) - center[1]
        interpolators = [RGI((x, y), image[:,:,i], bounds_error=False, fill_value=0) for i in range(3)]
        
        # indices we want to transform to (the warped version)
        initial_indices = np.where(image[:,:,0] > -1)

        # indices in the original image we want to look at, and interpolate into
        indices_to_obtain = mapping(initial_indices, focal_length, center, scaling)

        for channel in range(image.shape[2]):
            result[:,:,channel] = interpolators[channel](indices_to_obtain).reshape(image.shape[:2])
        return result


    # focal_length in pixels
    def runAlgorithm(self, folder_name, focal_length, mapping):
        img_names = glob.glob("data/" + folder_name + "/*")
        for img_name in img_names:
            image = readimage(img_name)
            mapped = self.warpImage(image, focal_length, mapping)
            publishImage(mapped)
        return

    def runMain(self):
        print "\nRunning Cylindrical Panorama Algorithm:"
        self.runAlgorithm("synthetic", 1320, self.cylindricalMappingIndices)
        self.runAlgorithm("synthetic", 1320, self.sphericalMappingIndices)
        return





# Main
if __name__ == "__main__":
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    Panorama().runMain()
    print ""



