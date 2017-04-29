#!/usr/bin/env python
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import sys, os, time, glob
from scipy import misc
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.ndimage.filters import gaussian_filter, sobel
from scipy.signal import fftconvolve
from scipy.sparse.linalg import cg
from scipy.linalg import cho_solve, cho_factor, cholesky
import pyamg

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



def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
def compose(source, dest, shift, h):
    #bounding box
    top_left_x = min(0, shift)
    top_left_y = min(0, h)
    bottom_right_x = max(dest.shape[1], shift + source.shape[1])
    bottom_right_y = max(dest.shape[0], h + source.shape[0])

    total_img = None
    if source.ndim > 2:
        total_img = np.zeros((bottom_right_y - top_left_y, bottom_right_x - top_left_x, source.shape[2]))
    else:
        total_img = np.zeros((bottom_right_y - top_left_y, bottom_right_x - top_left_x))

    #insert destination image
    dest_start_x = max(0, -shift)
    dest_start_y = max(0, -h)
    total_img[dest_start_y:dest_start_y + dest.shape[0], dest_start_x:dest_start_x + dest.shape[1]] = dest

    #insert source image
    source_start_x = max(0, shift)
    source_start_y = max(0, h)
    total_img[source_start_y:source_start_y + source.shape[0], source_start_x:source_start_x + source.shape[1]] = source
    return total_img
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
    def convolve(self, source, dest, method="pyramid", freq_type="hpf"):
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
        elif method == "fourier":
            h, shift = self.fourier_convolve(source, dest)
        else:
            raise ValueError("Method not recognized.")
        return h, shift, compose(dest, source, shift, h)
    def calcImagePyramid(self, img, threshold=30):
        imgs = []
        current = img
        imgs.insert(0, current)
        while current.shape[0] > threshold and current.shape[1] > threshold:
            current = misc.imresize(current, 0.5)
            imgs.insert(0, current)
        return imgs
    def pyramid_convolve_gradient(self, source, dest, threshold=30, drift_min=-20, drift_max=20, shift_min = 0, shift_max = None):
        source_x = sobel(source, axis=0, mode="constant")
        source_y = sobel(source, axis=1, mode="constant")
        source = np.hypot(source_x, source_y)

        dest_x = sobel(dest, axis=0, mode="constant")
        dest_y = sobel(dest, axis=1, mode="constant")
        dest = np.hypot(dest_x, dest_y)
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
    def fourier_convolve(self, source, dest):
        source_img = rgb2gray(source)
        dest_img = rgb2gray(dest)

        source_x = sobel(source_img, axis=0, mode="constant")
        source_y = sobel(source_img, axis=1, mode="constant")
        source_img = np.hypot(source_x, source_y)

        dest_x = sobel(dest_img, axis=0, mode="constant")
        dest_y = sobel(dest_img, axis=1, mode="constant")
        dest_img = np.hypot(dest_x, dest_y)
        displayImage(dest_img)
        displayImage(source_img)
        ndims = (source_img.shape[0] + dest_img.shape[0] - 1, source_img.shape[1] + dest_img.shape[1] - 1)
        source_freq = np.conjugate(np.fft.fft2(source_img, ndims))
        dest_freq = np.fft.fft2(dest_img, ndims)
        total_freq = np.multiply(source_freq, dest_freq)/np.absolute(np.multiply(source_freq, dest_freq))
        total_img = np.fft.ifft2(total_freq).real
        print(total_img.shape)
        displayImage(total_img)
  
        h, shift = np.unravel_index(np.argmax(total_img), total_img.shape)
        return h, shift
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
        return min_h, min_shift
        

    # focal_length in pixels
    def runAlgorithm(self, folder_name, focal_length, mapping):
        img_names = glob.glob("data/" + folder_name + "/*")  # TODO: Remove slicing after finished testing
        poisson = PoissonSolver()
        panorama = []
        image = readimage(img_names[0])
        mapped = self.warpImage(image, focal_length, mapping)
        mapped = misc.imresize(mapped, 0.5).astype(float)/255
        panorama = mapped
        for img_name in img_names[1:]:
            image = readimage(img_name)
            mapped = self.warpImage(image, focal_length, mapping)
            mapped = misc.imresize(mapped, 0.5).astype(float)/255
            print "Convolving now: "+time.ctime()  # TODO: Remove after tested
            h, shift, simple_panorama = self.convolve(panorama, mapped, method="pyramid")
            mask = np.ones((mapped.shape[0], mapped.shape[1]))
            print "Poisson:"
            #red
            print str(h) + ":" + str(shift)
            red = poisson.poisson(mapped[:,:,0], panorama[:,:,0], mask, (h, shift), poisson.seamless_gradient)
            #displayImage(red)
            #green
            green = poisson.poisson(mapped[:,:,1], panorama[:,:,1], mask, (h, shift), poisson.seamless_gradient)
            #blue
            blue = poisson.poisson(mapped[:,:,2], panorama[:,:,2], mask, (h, shift), poisson.seamless_gradient)

            panorama = np.zeros((red.shape[0], red.shape[1], 3))
            panorama[:,:,0] = red
            panorama[:,:,1] = green
            panorama[:,:,2] = blue
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

class PoissonSolver:
    def gauss_seidel(self, A, b, iterations=1000):
        x = np.zeros(b.shape)
        for i in range(iterations):
            x_n = np.zeros(b.shape)
            for i in range(len(A)):
                s1 = np.dot(A[i, :i], x_n[:i])
                s2 = np.dot(A[i, i + 1:], x[i + 1:])
                x_n[i] = (b[i] - s1 - s2) / A[i, i]

            if np.allclose(x, x_n, rtol=1e-8):
                break
            x = x_n
        return x
    def seamless_gradient(self, src, dst, start, end, point_tl):
        start_val = 0
        if start[0] >= 0 and start[0] < src.shape[0] and start[1] >= 0 and start[1] < src.shape[1]:
            start_val = src[start]
        end_val = start_val
        if end[0] >= 0 and end[0] < src.shape[0] and end[1] >= 0 and end[1] < src.shape[1]:
            end_val = src[end]
        return start_val - end_val
    def mixed_gradient(self, src, dst, start, end, point_tl):
        src_gradient = self.seamless_gradient(src, dst, start, end, point_tl)
        nstart = (start[0] + point_tl[0], start[1] + point_tl[1])
        nend = (end[0] + point_tl[0], end[1] + point_tl[1])
        dest_gradient = self.seamless_gradient(dst, src, nstart, nend, point_tl)
        if np.abs(dest_gradient) > np.abs(src_gradient):
            return dest_gradient
        else:
            return src_gradient
    def poisson(self, src, dst, mask, point_tl, guidance_func):
        region = mask.shape
        num_vertices = region[0] * region[1]

        A = np.zeros((num_vertices, num_vertices))
        b = np.zeros(num_vertices)
        #Build matrix
        for y in range(region[0]):
            for x in range(region[1]):
                x_dst = x + point_tl[1]
                y_dst = y + point_tl[0]
                if mask[y, x] > 0.5:               
                    #index
                    i = x + y * region[1]
                    counter = 0

                    b[i] += guidance_func(src, dst, (y, x), (y, x+1), point_tl)
                    if x + 1 < region[1] and mask[y, x + 1] > 0.5:
                        A[i,i + 1] = -1
                        counter += 1
                    else:
                        y_neighbor = y + point_tl[0]
                        x_neighbor = x + 1 + point_tl[1]
                        if x_neighbor < dst.shape[1] and x_neighbor >= 0 and y_neighbor < dst.shape[0] and y_neighbor >= 0:
                            counter += 1
                            b[i] += dst[y_neighbor, x_neighbor]
                        elif x + 1 < region[1]:
                            counter += 1
                            b[i] += src[y, x+1]
                        else:
                            counter += 1
                            b[i] += src[y, x]

                    b[i] += guidance_func(src, dst, (y, x), (y, x-1), point_tl)
                    if x - 1 >= 0 and mask[y, x - 1] > 0.5:
                        A[i,i - 1] = -1
                        counter += 1
                    else:
                        y_neighbor = y + point_tl[0]
                        x_neighbor = x - 1 + point_tl[1]
                        if x_neighbor < dst.shape[1] and x_neighbor >= 0 and y_neighbor < dst.shape[0] and y_neighbor >= 0:
                            counter += 1
                            b[i] += dst[y_neighbor, x_neighbor]
                        elif x - 1 >= 0:
                            counter += 1
                            b[i] += src[y, x-1]
                        else:
                            counter += 1
                            b[i] += src[y, x]

                    b[i] += guidance_func(src, dst, (y, x), (y + 1, x), point_tl)
                    if y + 1 < region[0] and mask[y + 1, x] > 0.5:
                        A[i, i + region[1]] = -1
                        counter += 1
                    else:
                        y_neighbor = y + 1 + point_tl[0]
                        x_neighbor = x + point_tl[1]
                        if x_neighbor < dst.shape[1] and x_neighbor >= 0 and y_neighbor < dst.shape[0] and y_neighbor >= 0:
                            counter += 1
                            b[i] += dst[y_neighbor, x_neighbor]
                        elif y + 1 < region[0]:
                            counter += 1
                            b[i] += src[y + 1, x]
                        else:
                            counter += 1
                            b[i] += src[y, x]
                    
                    b[i] += guidance_func(src, dst, (y, x), (y - 1, x), point_tl)
                    if y - 1 >= 0 and mask[y - 1, x] > 0.5:
                        A[i, i - region[1]] = -1
                        counter += 1
                    else:
                        y_neighbor = y - 1 + point_tl[0]
                        x_neighbor = x + point_tl[1]
                        if x_neighbor < dst.shape[1] and x_neighbor >= 0 and y_neighbor < dst.shape[0] and y_neighbor >= 0:
                            counter += 1
                            b[i] += dst[y_neighbor, x_neighbor]
                        elif y - 1 >= 0:
                            counter += 1
                            b[i] += src[y-1, x]
                        else:
                            counter += 1
                            b[i] += src[y, x]
                    A[i,i] = counter
        print A.shape
        #c_factors = cho_factor(A)
        #points = cho_solve(c_factors, b)
        #points = cg(A, b)[0]
        #points = self.gauss_seidel(A, b, 5000)
        points = pyamg.solve(A, b, verb=False)
        print points
        img = np.zeros(region)
        for y in range(region[0]):
            for x in range(region[1]):
                if mask[y, x] > 0.5:
                    #index
                    i = x + y * region[1]
                    if points[i] > 1:
                        img[y, x] = 1
                    elif points[i] < 0:
                        img[y, x] = 0
                    else:
                        img[y, x] = points[i]

        return compose(img, dst, point_tl[1], point_tl[0])

# Main
if __name__ == "__main__":
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    Panorama().runMain()
    print ""
    ### Code to test convolve(): ###
    #source = readimage("data/o-brienleft.jpg")
    #dest = readimage("data/o-brienright.jpg")
    #panorama = Panorama()
    #total_img = panorama.convolve(source, dest, method="pyramid")
    #print np.max(total_img)
    #displayImage(total_img)
    ### Code to test Poisson(): ###
    #print "Poisson now: "+time.ctime()  # TODO: Remove after tested
    #dest = rgb2gray(readimage("output/image1.png"))
    #source = rgb2gray(readimage("output/image2.png"))
    #dest = misc.imresize(dest, 0.5)
    #source = misc.imresize(source, 0.5)
    #quarter_x = int(231/2)
    #quarter_y = 0
    #mask = np.ones(source.shape)
    #poisson = PoissonSolver()
    #dest = dest.astype(float)/255
    #source= source.astype(float)/255
    #output_img = poisson.poisson(dest, source, mask, (quarter_y, quarter_x), poisson.seamless_gradient)
    #displayImage(output_img)
    #print "Done poisson: "+time.ctime()  # TODO: Remove after tested


