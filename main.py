#!/usr/bin/env python
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import IPython



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

    def runAlgorithm(self, folder_name):
        return

    def runMain(self):
        print "\nRunning Cylindrical Panorama Algorithm:"
        self.runAlgorithm("scene1")
        return





# Main
if __name__ == "__main__":
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    Panorama().runMain()
    print ""



