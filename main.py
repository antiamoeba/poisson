#!/usr/bin/env python
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import sys, os
from scipy import misc



# Parameters and defaults
image_num = 1



def readimage(name):
    image = misc.imread(name)
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
    plt.show()
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

class Convolution:
    def convolve(self, source, dest, drift_max, shift_min = 0, shift_max = None):
        if shift_max == None:
            shift_max = source.shape[1] - 1
        min_conv = np.inf
        min_h = np.inf
        min_shift = np.inf
        lo = None
        ro = None
        for drift in range(-drift_max, drift_max):
            for start in range(shift_min, shift_max):
                end = start + dest.shape[1]
                if end > source.shape[1]:
                    end = source.shape[1]
                source_start = max(0, start)
                dest_start = max(0, -start)

                source_overlap = source[max(drift,0):min(drift + source.shape[0],source.shape[0]),source_start:end]

                dest_overlap = dest[max(-drift, 0):min(-drift + dest.shape[0],dest.shape[0]),dest_start:end-start]

                total = np.mean((source_overlap - dest_overlap)**2)
                if total < min_conv:
                    min_conv = total
                    min_shift = start
                    min_h = drift
                    lo = source_overlap
                    ro = dest_overlap
        total_width = min_shift + dest.shape[1]
        total_height = abs(min_h) + source.shape[0]
        total_img = np.zeros((total_height, total_width, 3))
        
        start_left_h = max(-min_h, 0)
        total_img[start_left_h:start_left_h + source.shape[0],:min_shift] = source[:,:min_shift]
        start_right_h = max(min_h, 0)

        total_img[start_right_h:start_right_h + dest.shape[0],min_shift:] = dest
        total_img = total_img.astype(np.uint8)
        return min_h, min_shift, total_img
        

# Main
if __name__ == "__main__":
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    Panorama().runMain()
    print ""
    source = readimage("data/o-brienleft.png")[:-10]
    dest = readimage("data/o-brienright.png")[10:]
    min_h, min_shift, total_img = Convolution().convolve(source, dest, 20)
    displayImage(total_img)


