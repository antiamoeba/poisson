#!/usr/bin/env python
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import sys, os, time, glob
from scipy import misc
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.ndimage.filters import gaussian_filter, sobel

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

        slice_edge = int(0.08 * result.shape[1] / 2.0)
        result = result[:,slice_edge:-slice_edge,:]
        return result
    def compose(self, source, dest, shift, h):
        total_width = shift + dest.shape[1]
        total_height = abs(h) + source.shape[0]
        total_img = np.zeros((total_height, total_width, 3))
        
        start_left_h = max(-h, 0)
        total_img[start_left_h:start_left_h + source.shape[0],:shift] = source[:,:shift]
        start_right_h = max(h, 0)

        total_img[start_right_h:start_right_h + dest.shape[0],shift:] = dest
        #total_img = total_img.astype(np.uint8)
        return total_img
    def convolve(self, source, dest, method="pyramid", freq_type="hpf"):
        print "source:" + str(source.shape)
        print "dest:" + str(dest.shape)
        if method == "pyramid":
            h, shift = self.pyramid_convolve(source, dest)
        elif method == "raw":
            h, shift = self.raw_convolve(source, dest)
        elif method == "gradient":
            h, shift = self.pyramid_convolve_gradient(source, dest)
        elif method == "freq":
            h, shift = self.freq_convolve(source, dest, freq_type)
        elif method == "freqspace":
            h, shift = self.freqspace_convolve(source, dest, freq_type)
        else:
            raise ValueError("Method not recognized.")
        return self.compose(source, dest, shift, h)
    def calcImagePyramid(self, img, threshold=30):
        imgs = []
        current = img
        imgs.insert(0, current)
        while current.shape[0] > threshold and current.shape[1] > threshold:
            current = misc.imresize(current, 0.5)
            imgs.insert(0, current)
        return imgs
    def pyramid_convolve_gradient(self, source, dest, threshold=30, drift_min=-20, drift_max=20, shift_min = 0, shift_max = None):
        source = sobel(source)
        dest = sobel(dest)
        return self.pyramid_convolve(source, dest, threshold, drift_min, drift_max, shift_min, shift_max)
    def pyramid_convolve(self, source, dest, threshold=30, drift_min=-20, drift_max=20, shift_min = 0, shift_max = None):
        #if source.shape != dest.shape:
        #    raise ValueError('Source and destination images do not have the same dimension!')
        sourcePyramid = self.calcImagePyramid(source, threshold)
        destPyramid = self.calcImagePyramid(dest, threshold)
        
        #start
        if shift_max == None:
            shift_max = source.shape[1] - 1
        start_drift_min = int(np.ceil(drift_min/(2 ** (len(destPyramid)-1))))
        start_drift_max = int(np.ceil(drift_max/(2 ** (len(destPyramid)-1))))
        start_shift_min = int(np.ceil(shift_min/(2 ** (len(destPyramid)-1))))
        start_shift_max = int(np.ceil(shift_max/(2 ** (len(destPyramid)-1))))
        curr_h, curr_shift = self.raw_convolve(sourcePyramid[0], destPyramid[0], start_drift_min, start_drift_max, start_shift_min, start_shift_max)

        counter = 1
        while counter < len(destPyramid):
            curr_source = sourcePyramid[counter]
            curr_dest = destPyramid[counter]
            curr_h, curr_shift = self.raw_convolve(curr_source, curr_dest, curr_h*2 - 2, curr_h*2 + 2, curr_shift*2 - 2, curr_shift*2 + 2)
            counter += 1
        return curr_h, curr_shift
    def getPower2Dims(self, dims):
        return (2**np.ceil(np.log2(dims))).astype(int)
    def rgb2gray(self, rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    def freq_convolve(self, source, dest, filter_type="lpf"):
        source_img = self.rgb2gray(source)
        dest_img = self.rgb2gray(dest)
        
        source_dims = self.getPower2Dims(source_img.shape)
        dest_dims = self.getPower2Dims(dest_img.shape)

        source_freq = np.fft.fft2(source_img, axes=(0, 1))
        dest_freq = np.fft.fft2(dest_img, axes=(0, 1))

        if filter_type == "hpf":
            source_freq[:int(source_freq.shape[0]/5),:int(source_freq.shape[1]/5)] = 0
            dest_freq[:int(dest_freq.shape[0]/5),:int(dest_freq.shape[1]/5)] = 0
        elif filter_type == "lpf":
            source_freq[int(source_freq.shape[0]/5):,int(source_freq.shape[1]/5):] = 0
            dest_freq[int(dest_freq.shape[0]/5):,int(dest_freq.shape[1]/5):] = 0
        else:
            raise ValueError("Filter type not recognized!")

        source_out = np.fft.ifft2(source_freq).real
        dest_out = np.fft.ifft2(dest_freq).real
        h, shift = self.raw_convolve(source_out, dest_out)
        print str(h) + ":" + str(shift)
        return h, shift
    def freqspace_convolve(self, source, dest, filter_type="hpf"):
        source_img = self.rgb2gray(source)
        dest_img = self.rgb2gray(dest)
        source_freq = None
        dest_freq = None
        if filter_type == "lpf":
            source_freq = gaussian_filter(source_img, (source_img.shape[0]/5, source_img.shape[1]/5))
            dest_freq = gaussian_filter(dest_img, (dest_img.shape[0]/5, dest_img.shape[1]/5))
        elif filter_type == "hpf":
            source_im = gaussian_filter(source_img, (source_img.shape[0]/5, source_img.shape[1]/5))
            dest_im = gaussian_filter(dest_img, (dest_img.shape[0]/5, dest_img.shape[1]/5))
            source_freq = source_img - source_im
            dest_freq = dest_img - dest_im
        else:
            raise ValueError("Filter type not recognized!")

        h, shift = self.raw_convolve(source_freq, dest_freq)
        print str(h) + ":" + str(shift)
        return h, shift
    def raw_convolve(self, source, dest, drift_min=-20, drift_max=20, shift_min = 0, shift_max = None):
        if shift_max == None:
            shift_max = source.shape[1] - 1
        min_conv = np.inf
        min_h = np.inf
        min_shift = np.inf
        for drift in range(drift_min, drift_max + 1):
            for start in range(shift_min, min(shift_max, source.shape[1])):
                end = start + dest.shape[1]
                if end > source.shape[1]:
                    end = source.shape[1]
                source_start = max(0, start)
                dest_start = max(0, -start)

                source_overlap = source[max(drift,0):min(drift + dest.shape[0],source.shape[0]),source_start:end]

                dest_overlap = dest[max(-drift, 0):min(-drift + source.shape[0],dest.shape[0]),dest_start:end-start]
                total = np.mean(np.absolute(source_overlap - dest_overlap)**2)
                if total < min_conv:
                    min_conv = total
                    min_shift = start
                    min_h = drift
        print min_conv
        return min_h, min_shift
        

    # focal_length in pixels
    def runAlgorithm(self, folder_name, focal_length, mapping):
        img_names = glob.glob("data/" + folder_name + "/*")[:2]  # TODO: Remove slicing after finished testing
        panorama = []
        image = readimage(img_names[0])
        mapped = self.warpImage(image, focal_length, mapping)
        panorama = mapped
        for img_name in img_names[1:]:
            image = readimage(img_name)
            mapped = self.warpImage(image, focal_length, mapping)
            print "Convolving now: "+time.ctime()  # TODO: Remove after tested
            panorama = self.convolve(panorama, mapped, method="pyramid_gradient")
            print "Panorama:" + str(panorama.shape)
            print "Done convolving: "+time.ctime()  # TODO: Remove after tested
        publishImage(panorama)
        return panorama


    def runMain(self):
        print "\nRunning Cylindrical Panorama Algorithm:"
        self.runAlgorithm("synthetic", 330, self.cylindricalMappingIndices)

        # print "\nRunning Spherical Panorama Algorithm:"
        # self.runAlgorithm("synthetic", 1320, self.sphericalMappingIndices)
        return


        

# Main
if __name__ == "__main__":
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    #Panorama().runMain()
    #print ""
    ### Code to test convolve(): ###
    source = readimage("data/Synthetic/img01.jpg")
    dest = readimage("data/Synthetic/img09.jpg")
    panorama = Panorama()
    source = panorama.warpImage(source,330, panorama.cylindricalMappingIndices)
    dest = panorama.warpImage(dest, 330, panorama.cylindricalMappingIndices)
    total_img = panorama.convolve(source, dest, method="freq", freq_type="lpf")
    print np.max(total_img)
    displayImage(total_img)


