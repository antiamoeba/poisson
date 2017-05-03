#!/usr/bin/env python
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import sys, os, time, glob
import scipy
from scipy import misc
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.ndimage.filters import gaussian_filter, sobel, laplace
from scipy.ndimage.morphology import binary_erosion
from scipy.signal import fftconvolve
from scipy.sparse.linalg import cg
from scipy import sparse
from scipy.linalg import cho_solve, cho_factor, cholesky
from skimage.draw import circle
from skimage.feature import corner_harris, peak_local_max
from skimage import img_as_float
from skimage.transform import resize
import pyamg

# Parameters and defaults
image_num = 1



def readimage(name):
    image = misc.imread(name)
    if name[-4:] == ".jpg":
        image = image
    elif name[-4:] == ".png":
        image = image[:,:,:3]
    return img_as_float(image)

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

def publishImageScatter(image, scatter1, scatter2=None, scatter3=None, scatter4=None, title=None):
    if len(image.shape) == 2:
        image = np.dstack([image for _ in range(3)])
    if title:
        imgName = output_dir + '/' + title
    else:
        global image_num
        imgName = output_dir + '/image' + str(image_num) + '.png'
        image_num += 1
    plt.figure()
    plt.axis('off')
    plt.imshow(image)
    plt.scatter(x=scatter1[0], y=scatter1[1], c='r', s=15)
    if scatter2 is not None:
        plt.scatter(x=scatter2[0], y=scatter2[1], c='b', s=15)
    if scatter3 is not None:
        plt.scatter(x=scatter3[0], y=scatter3[1], c='g', s=15)
    if scatter4 is not None:
        plt.scatter(x=scatter4[0], y=scatter4[1], c='y', s=15)
    plt.savefig(imgName, bbox_inches='tight')
    plt.close()
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

def compose(source, dest, shift, h, start=False):
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
    if start == True:
        return total_img, source_start_y, source_start_x
    else:
        return total_img
def mask_compose(source, dest, mask, shift, h, start=False):
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

    #insert source image with mask
    source_start_x = max(0, shift)
    source_start_y = max(0, h)
    total_img[source_start_y:source_start_y + source.shape[0], source_start_x:source_start_x + source.shape[1]] *= 1 - mask
    total_img[source_start_y:source_start_y + source.shape[0], source_start_x:source_start_x + source.shape[1]] += np.multiply(mask, source)
    if start == True:
        return total_img, source_start_y, source_start_x
    else:
        return total_img

def calcImagePyramid(img, threshold=30, levels=None, resize="nearest"):
    imgs = []
    current = img_as_float(img)
    imgs.insert(0, current)
    if levels == None:
        while current.shape[0] > threshold and current.shape[1] > threshold:
            current = img_as_float(misc.imresize(current, 0.5))
            imgs.insert(0, current)
    else:
        iterations = 1
        while iterations < levels:
            if np.all(current == 1):
                current = np.ones((int(current.shape[0]/2),int(current.shape[1]/2)))
            else:
                current = img_as_float(misc.imresize(current, 0.5, interp=resize))
            imgs.insert(0, current)
            iterations += 1
    return imgs
    
cone_height = 4
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
    
    def coneMappingIndices(self, initial_indices, focal_length, center, scaling):
        new_x = focal_length * np.tan((initial_indices[1] - center[1]) / scaling)
        w = np.sqrt(focal_length**2 + new_x**2)
        new_y = cone_height * w * (initial_indices[0] - center[0]) / (cone_height * scaling - (initial_indices[0] - center[0]))
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
        sourcePyramid = calcImagePyramid(source, threshold)
        destPyramid = calcImagePyramid(dest, threshold)
        
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
        ndims = (source_img.shape[0] + dest_img.shape[0] - 1, source_img.shape[1] + dest_img.shape[1] - 1)
        source_freq = np.conjugate(np.fft.fft2(source_img, ndims))
        dest_freq = np.fft.fft2(dest_img, ndims)
        total_freq = np.multiply(source_freq, dest_freq)/np.absolute(np.multiply(source_freq, dest_freq))
        total_img = np.fft.ifft2(total_freq).real
  
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
    
    def calcGuassianPyramid(self, img, max_levels=10, sigma=1):
        curr_img = img
        pyramid = [img]
        for i in range(max_levels - 1):
            curr_img = img_as_float(gaussian_filter(curr_img, sigma))
            pyramid.insert(0, curr_img)
        return pyramid

    def calcLaplacianPyramid(self, img, max_levels=10, sigma=1):
        curr_img = img
        pyramid = []
        for i in range(max_levels - 1):
            n_img = img_as_float(gaussian_filter(curr_img, sigma))
            curr_laplace = curr_img - n_img
            pyramid.insert(0, curr_laplace)
            curr_img = n_img
        pyramid.insert(0, curr_img)
        return pyramid

    def pyramid_blend(self, src, dest, h, shift, mask_type="flat", levels=10):
        src_pyramid = self.calcLaplacianPyramid(src, max_levels=levels)
        dest_pyramid = self.calcLaplacianPyramid(dest, max_levels=levels)
        total_pyramid = []
        
        #calculate overlap area in src coords
        src_br = (src.shape[0], src.shape[1])
        dest_br = (dest.shape[0] - h, dest.shape[1] - shift)
        dest_tl = (-h, -shift)
        overlap_br = (min(src_br[0], dest_br[0]), min(src_br[1], dest_br[1]))
        overlap_tl = (max(dest_tl[0], 0), max(dest_tl[1], 0))

        #build mask
        mask = None
        if mask_type == "flat":
            half_y = int((overlap_br[0] - overlap_tl[0])/2)
            half_x = int((overlap_br[1] - overlap_tl[1])/2)
            mask = np.ones(src.shape)
            mask[:,:overlap_tl[1] + half_x] = 0.0
            # mask[:,overlap_tl[1] + half_x:overlap_br[1]] = 0.85
        mask_pyramid = self.calcGuassianPyramid(mask, max_levels=levels)
        
        for src_img, dest_img, mask_img in zip(src_pyramid, dest_pyramid, mask_pyramid):
            total_img = mask_compose(src_img, dest_img, mask_img, shift, h)
            total_pyramid.append(total_img)
        
        final_img = total_pyramid[0]
        for i in range(1, len(total_pyramid)):
            final_img += total_pyramid[i]
        final_img[final_img > 1] = 1
        final_img[final_img < 0] = 0
        return final_img
        
    def poisson_pyramid(self, src, dest, h, shift, max_levels=2, method="seamless"):
        mask = np.ones((src.shape[0], src.shape[1]))
        #for y in range(src.shape[0]):
        #    for x in range(src.shape[1]):
        #        if src[y, x, 0] == 0 and src[y, x, 1] == 0 and src[y, x, 2] == 0:
        #            mask[y, x] = 0
        mask_pyramid = calcImagePyramid(mask, levels=max_levels)
        poisson = PoissonSolver()
        colors = []
        for i in range(3):
            src_curr = src[:,:,i]
            dest_curr = dest[:,:,i]
            src_pyramid = calcImagePyramid(src_curr, levels=max_levels)
            dest_pyramid = calcImagePyramid(dest_curr, levels=max_levels)
            prev_level = None
            if method == "seamless":
                prev_level = poisson.poisson(src_pyramid[0], dest_pyramid[0], mask_pyramid[0], (int(h/(2 ** (max_levels - 1))), int(shift/(2 ** (max_levels - 1)))), poisson.seamless_gradient, return_type="region")
            elif method == "mixed":
                prev_level = poisson.poisson(src_pyramid[0], dest_pyramid[0], mask_pyramid[0], (int(h/(2 ** (max_levels - 1))), int(shift/(2 ** (max_levels - 1)))), poisson.mixed_gradient, return_type="region")
            total_img = None
            for j in range(1, max_levels):
                h_offset = int(h/(2**((max_levels - 1)-j)))
                shift_offset = int(shift/(2**((max_levels - 1)-j)))
                mask_j = mask_pyramid[j]
                mask_j[mask_j >= 0.5] = 1
                mask_j[mask_j < 0.5] = 0
                prev_level = resize(prev_level, (mask_j.shape[0], mask_j.shape[1]))
                old_total, h_n, shift_n = mask_compose(prev_level, dest_pyramid[j], mask_j, shift_offset, h_offset, start=True)
                #erode mask
                slices = []
                curr_eroded = mask_j.astype(int)
                flag = False
                while not flag:
                    n_eroded = binary_erosion(curr_eroded, structure=np.ones((10 * j, 10 * j))).astype(int)
                    diff = curr_eroded - n_eroded
                    slices.append(diff)
                    curr_eroded = n_eroded
                    if np.count_nonzero(curr_eroded) == 0:
                        flag = True
                output_region = np.zeros(mask_j.shape)
                for slice_mask in slices:
                    slice_points = None
                    if method == "seamless":
                        slice_points = poisson.poisson(src_pyramid[j], old_total, slice_mask, (h_n, shift_n), poisson.seamless_gradient, return_type="region")
                    elif method == "mixed":
                        slice_points = poisson.poisson(src_pyramid[j], old_total, slice_mask, (h_n, shift_n), poisson.mixed_gradient, return_type="region")
                    output_region += slice_points
                prev_level = output_region
                total_img = mask_compose(output_region, dest_pyramid[j], mask_j, shift_offset, h_offset)
            colors.append(total_img)
        output_image = np.zeros((colors[0].shape[0], colors[0].shape[1], 3))
        for i in range(3):
            output_image[:,:,i] = colors[i]
        #displayImage(output_image)
        return output_image
        
    # focal_length in pixels
    def runAlgorithm(self, folder_name, focal_length, mapping, align_method="convolve", blend_method="poisson"):
        img_names = glob.glob("data/" + folder_name + "/*")[::-1]
        poisson = PoissonSolver()
        feature_detector = FeatureDetection()
        panorama = []
        image = readimage(img_names[0])
        mapped = self.warpImage(image, focal_length, mapping)[80:-80]
        mapped = img_as_float(misc.imresize(mapped, 0.25))
        panorama = mapped
        imgs = []
        start_panorama = panorama
        offsets = []
        print "Matching images together"
        for img_name in img_names[1:]:
            print img_name
            image = readimage(img_name)
            mapped = self.warpImage(image, focal_length, mapping)[80:-80]
            mapped = img_as_float(misc.imresize(mapped, 0.25))
            imgs.append(mapped)
            if align_method == "convolve":
                print "Convolving now: "+time.ctime()  # TODO: Remove after tested
                h, shift, panorama = self.convolve(panorama, mapped, method="gradient")
                offsets.append((h, shift))
                print (h, shift)
                print "Done convolving: "+time.ctime()  # TODO: Remove after tested
            elif align_method == "features":
                h, shift, correspondences = feature_detector.getAutoCorrespondences(panorama, mapped)
                panorama = compose(mapped, panorama, shift, h)
                start_one = max(0, h)
                start_two = max(0, -h)
                end_one = start_one + mapped.shape[0]
                end_two = start_two + panorama.shape[0]
                panorama = panorama[max(start_one, start_two):min(end_one, end_two)]
                offsets.append((h, shift))
            else:
                raise ValueError("Align method not recognized.")
        #Blend
        panorama = start_panorama
        offsets = np.array(offsets)
        cropIndex = feature_detector.cropPanoramaToWrap(imgs[np.argmin([0] + offsets[:,1])], imgs[np.argmax(offsets[:,1])], np.max(offsets[:,1]))
        print "Blending"
        for img, offset in zip(imgs, offsets):
            if blend_method == "poisson":
                panorama = self.poisson_pyramid(img, panorama, offset[0], offset[1])
            elif blend_method == "pyramid":
                panorama = self.pyramid_blend(img, panorama, offset[0], offset[1], levels=100)
            else:
                raise ValueError("Blend method not recognized")
            start_one = max(0, offset[0])
            start_two = max(0, -offset[0])
            end_one = start_one + img.shape[0]
            end_two = start_two + panorama.shape[0]
            panorama = panorama[max(start_one, start_two):min(end_one, end_two)]
        publishImage(panorama)
        panorama = panorama[:,:cropIndex]
        
        # Blending ends of the panorama
        publishImage(panorama)
        
        left_end = panorama[:,:200]
        right_end = panorama[:,-200:]
        if blend_method == "poisson":
            ends = self.poisson_pyramid(right_end, left_end, 0, right_end.shape[1])
        elif blend_method == "pyramid":
            ends = self.pyramid_blend(right_end, left_end, 0, right_end.shape[1], levels=100)
        panorama[:,:200] = left_end = ends[:,:200]
        panorama[:,-200:] = right_end = ends[:,-200:]

        publishImage(panorama)
        return panorama

    def runMain(self):
        print "\nRunning Cylindrical Panorama Algorithm:"
        # self.runAlgorithm("synthetic", 330, self.cylindricalMappingIndices)
        # self.runAlgorithm("vlsb", 6600.838, self.cylindricalMappingIndices)
        # self.runAlgorithm("woods", 6600.838, self.cylindricalMappingIndices)
        # self.runAlgorithm("vlsb", 1170, self.cylindricalMappingIndices, "features")
        self.runAlgorithm("vlsb", 1167, self.cylindricalMappingIndices, "features", "pyramid")
        # self.runAlgorithm("vlsb", 1167, self.coneMappingIndices, "features", "pyramid")

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
        if start[0] >= 0 and start[0] < src.shape[0] and start[1] >= 0 and start[1] < src.shape[1] and end[0] >= 0 and end[0] < src.shape[0] and end[1] >= 0 and end[1] < src.shape[1]:
            start_val = src[start]
            end_val = src[end]
            return start_val - end_val
        else:
            return 0
    def mixed_gradient(self, src, dst, start, end, point_tl):
        src_gradient = self.seamless_gradient(src, dst, start, end, point_tl)
        nstart = (start[0] + point_tl[0], start[1] + point_tl[1])
        nend = (end[0] + point_tl[0], end[1] + point_tl[1])
        
        dest_gradient = 0
        if nstart[0] >= 0 and nstart[0] < dst.shape[0] and nstart[1] >= 0 and nstart[1] < dst.shape[1] and nend[0] >= 0 and nend[0] < dst.shape[0] and nend[1] >= 0 and nend[1] < dst.shape[1]:
            end_val = dst[nend]
            start_val = dst[nstart]
            dest_gradient = start_val - end_val
        if abs(dest_gradient) > abs(src_gradient):
            return dest_gradient
        else:
            return src_gradient
    def poisson(self, src, dst, mask, point_tl, guidance_func, return_type="composed"):
        if mask==None:
            mask = np.ones(src.shape)
        mask[mask < 0.5] = 0
        mask[mask > 0.5] = 1
        region = mask.shape
        num_vertices = np.count_nonzero(mask)

        A = np.zeros((num_vertices, num_vertices))
        b = np.zeros(num_vertices)
        i = -1
        vertices = dict()
        #Build hashmap
        for y in range(region[0]):
            for x in range(region[1]):
                if mask[y, x] > 0.5:
                    i += 1
                    vertices[(y, x)] = i
        #Build matrix
        for y in range(region[0]):
            for x in range(region[1]):
                if mask[y, x] > 0.5:    
                    x_dst = x + point_tl[1]
                    y_dst = y + point_tl[0]           
                    #index
                    i = vertices[(y, x)]
                    #right
                    b[i] += guidance_func(src, dst, (y, x), (y, x+1), point_tl)
                    if x + 1 < region[1] and mask[y, x + 1] > 0.5:
                        right = vertices[(y, x+1)]
                        A[i,right] = -1
                    else:
                        y_neighbor = y + point_tl[0]
                        x_neighbor = x + 1 + point_tl[1]
                        if x_neighbor < dst.shape[1] and x_neighbor >= 0 and y_neighbor < dst.shape[0] and y_neighbor >= 0:
                            b[i] += dst[y_neighbor, x_neighbor]
                        else:
                            b[i] += src[y, x]
                    
                    #left
                    b[i] += guidance_func(src, dst, (y, x), (y, x-1), point_tl)
                    if x - 1 >= 0 and mask[y, x - 1] > 0.5:
                        left = vertices[(y, x-1)]
                        A[i,left] = -1
                    else:
                        y_neighbor = y + point_tl[0]
                        x_neighbor = x - 1 + point_tl[1]
                        if x_neighbor < dst.shape[1] and x_neighbor >= 0 and y_neighbor < dst.shape[0] and y_neighbor >= 0:
                            b[i] += dst[y_neighbor, x_neighbor]
                        else:
                            b[i] += src[y, x]
                    
                    #bottom
                    b[i] += guidance_func(src, dst, (y, x), (y + 1, x), point_tl)
                    if y + 1 < region[0] and mask[y + 1, x] > 0.5:
                        bottom = vertices[(y+1, x)]
                        A[i, bottom] = -1
                    else:
                        y_neighbor = y + 1 + point_tl[0]
                        x_neighbor = x + point_tl[1]
                        if x_neighbor < dst.shape[1] and x_neighbor >= 0 and y_neighbor < dst.shape[0] and y_neighbor >= 0:
                            b[i] += dst[y_neighbor, x_neighbor]
                        else:
                            b[i] += src[y, x]
                        
                    
                    #top
                    b[i] += guidance_func(src, dst, (y, x), (y - 1, x), point_tl)
                    if y - 1 >= 0 and mask[y - 1, x] > 0.5:
                        top = vertices[(y-1, x)]
                        A[i, top] = -1
                    else:
                        y_neighbor = y - 1 + point_tl[0]
                        x_neighbor = x + point_tl[1]
                        if x_neighbor < dst.shape[1] and x_neighbor >= 0 and y_neighbor < dst.shape[0] and y_neighbor >= 0:
                            b[i] += dst[y_neighbor, x_neighbor]
                        else:
                            b[i] += src[y, x]
                    A[i,i] = 4
        A = sparse.csr_matrix(A)
        #c_factors = cho_factor(A)
        #points = cho_solve(c_factors, b)
        #points = cg(A, b)[0]
        #points = self.gauss_seidel(A, b, 5000)
        points = pyamg.solve(A, b, verb=False)
        if return_type == "raw":
            return points
        img = np.zeros(region)
        i = -1
        for y in range(region[0]):
            for x in range(region[1]):
                if mask[y, x] > 0.5:
                    #index
                    i += 1
                    if points[i] > 1:
                        img[y, x] = 1
                    elif points[i] < 0:
                        img[y, x] = 0
                    else:
                        img[y, x] = points[i]
        if return_type == "composed":
            return mask_compose(img, dst, mask, point_tl[1], point_tl[0])
        if return_type == "region":
            return img



class FeatureDetection:
    def __init__(self, plot=False):
        self.plot = plot
        return

    def get_harris_corners(self, im, edge_discard=20):
        assert edge_discard >= 20

        # find harris corners
        h = corner_harris(im, method='eps', sigma=1)
        coords = peak_local_max(h, min_distance=1, indices=True)

        # discard points on edge
        edge = edge_discard  # pixels
        mask = (coords[:, 0] > edge) & \
               (coords[:, 0] < im.shape[0] - edge) & \
               (coords[:, 1] > edge) & \
               (coords[:, 1] < im.shape[1] - edge)
        coords = coords[mask].T
        return h, coords

    def dist2(self, x, c):
        ndata, dimx = x.shape
        ncenters, dimc = c.shape
        assert (dimx == dimc), 'Data dimension does not match dimension of centers'

        return (np.ones((ncenters, 1)) * np.sum((x**2).T, axis=0)).T + \
                np.ones((   ndata, 1)) * np.sum((c**2).T, axis=0)    - \
                2 * np.inner(x, c)

    def getHarrisCorners(self, image):
        h, (ys, xs) = self.get_harris_corners(image)
        return h, np.array([xs, ys])
    
    def nonMaxSuppression(self, h, corners):
        def findMaxRadius(h, corner):
            radius = 15 # arbitrary starting point
            low = 2
            high = int(np.max(h.shape)*np.sqrt(2)/2 + 0.999)
            while True:
                rr, cc = circle(corner[1], corner[0], radius, shape=h.shape)
                arg = np.argmax(h[rr,cc])
                isMax = np.all((cc[arg], rr[arg]) == corner)
                if not isMax:
                    high = radius-1
                else:
                    low = radius
                radius = int((high+low)/2.0 + 0.5)
                if high == low:
                    break
            return radius
        rads = [findMaxRadius(h, corner) for corner in corners.T]
        self.numCorners = 500
        inds = np.argsort(rads)[::-1][:self.numCorners]
        return corners[:,inds]
    
    def extractFeatureDescriptors(self, image, corners):
        fds = np.array([scipy.misc.imresize(image[y-20:y+20, x-20:x+20], (8,8)) for x,y in corners.T])
        fds = fds.reshape((fds.shape[0], 64))
        fds = (fds.T - np.mean(fds, axis=1)).T
        std = np.std(fds, axis=1)
        std = [1 if s == 0 else s for s in std]
        fds = np.divide(fds.T, std).T
        fds = fds.reshape((fds.shape[0], 8, 8))
        return fds

    def plotFeatureDescriptor(self, fd):
        result = np.zeros((25*9, 20*9))
        for i in range(25):
            for j in range(20):
                result[i*9:i*9+8, j*9:j*9+8] = fd[i*20 + j]
        publishImage(result)
        return
    
    def featureMatching(self, desc1, desc2):
        pairs = [] # indices
        dists = self.dist2(desc1.reshape(desc1.shape[0], 64), desc2.reshape(desc2.shape[0], 64))
        inds = np.argsort(dists, axis=1)[:,:2]
        for i, indices in enumerate(inds):
            if dists[i, indices[0]]/dists[i, indices[1]] < 0.4:
                pairs.append((i, indices[0]))
        return np.array(pairs)

    def computeMinOffset(self, pts1, pts2):
        x1 = pts1[:,0]
        x2 = pts2[:,0]
        # solving quadratic for offset
        a = pts1.shape[0]
        b = 2 * np.sum(x2 - x1)
        # c = np.sum((x1 - x2)**2)
        offset_x = - b / (2*a)

        y1 = pts1[:,1]
        y2 = pts2[:,1]
        a = pts1.shape[0]
        b = 2 * np.sum(y2 - y1)
        offset_y = - b / (2*a)

        return offset_x, offset_y

    # choose 4 random point pairs
    # compute smallest indexing error for min x and y offset
    # using that offset, compute error for all point indices
    # return set of correspondences which has the smallest offset, and that offset
    def estimateRANSAC(self, im1, im2, corners1, corners2, repeat=100):
        # 4-pt RANSAC for correspondences
        best = []
        best_offset_x, best_offset_y = 0, 0
        inds = np.array(range(corners1.shape[1]))
        for _ in range(1000000):
            # choose 4 random pairs (corners are indices!)
                # Favors right-most features more
            sorted_indices = np.argsort(corners1[0])[::-1]
            sorted_indices = np.log(sorted_indices + 2)
            random_indices = np.random.choice(inds, size=4, replace=False, p=sorted_indices/np.sum(sorted_indices))
                # Favors all features equally:
            # np.random.shuffle(inds)
            im1_pts = corners1[:,random_indices[:4]]
            im2_pts = corners2[:,random_indices[:4]]
            # compute offset
            offset_x, offset_y = self.computeMinOffset(im1_pts.T, im2_pts.T)
            offset_x, offset_y = np.round(offset_x).astype(int), np.round(offset_y).astype(int)
            # compute inliers (differences of coordinate distances)
            diff_x = corners2[0] + offset_x - corners1[0]
            diff_y = corners2[1] + offset_y - corners1[1]
            distances = diff_x**2 + diff_y**2
            rr = np.where(distances < 4)
            # if # inliers > then size of best, keep set of inliers
            if len(rr) > len(best):
                best = inds[rr]
                best_offset_x, best_offset_y = offset_x, offset_y
                break
            if len(rr) == corners1.shape[1]:
                break
        
        if len(best) <= 4 and repeat > 0:
            return self.estimateRANSAC(im1, im2, corners1, corners2, repeat - 1)
        elif len(best) <= 4:
            if corners1.shape[1] > 2 and corners2.shape[1] > 2:
                offset_x, offset_y = self.computeMinOffset(corners1.T, corners2.T)
                offset_x, offset_y = np.round(offset_x).astype(int), np.round(offset_y).astype(int)
                print "\tUsing all points for correspondences"
                return offset_x, offset_y, corners1, corners2
            else:
                best_offset_x, best_offset_y = im1.shape[1], 0

        print "\tFound " + str(len(best)) + " correspondences"
        return best_offset_y, best_offset_x, corners1[:,best], corners2[:,best]

    def getAutoCorrespondences(self, im1, im2):
        # Detecting Harris Corners...
        grayscale1 = rgb2gray(im1)
        grayscale2 = rgb2gray(im2)
        h1, corners1 = self.getHarrisCorners(grayscale1)
        h2, corners2 = self.getHarrisCorners(grayscale2)
        if self.plot:
            publishImageScatter(im1, corners1)
            publishImageScatter(im2, corners2)

        # Running Adaptive Non-Maximal Suppression...
        corners3 = self.nonMaxSuppression(h1, corners1)
        corners4 = self.nonMaxSuppression(h2, corners2)
        if self.plot:
            publishImageScatter(im1, corners3)
            publishImageScatter(im2, corners4)
            publishImageScatter(im1, corners1, corners3)
            publishImageScatter(im2, corners2, corners4)

        # Extracting Feature Descriptors...
        fd1 = self.extractFeatureDescriptors(grayscale1, corners3)
        fd2 = self.extractFeatureDescriptors(grayscale2, corners4)
        if self.plot:
            self.plotFeatureDescriptor(fd1)
            self.plotFeatureDescriptor(fd2)

        # Running Feature Matching...
        pairs = self.featureMatching(fd1, fd2)
        if len(pairs) == 0:
            print "No matching features found!"
            sys.exit(1)
        corners5 = corners3[:, pairs[:,0]]
        corners6 = corners4[:, pairs[:,1]]
        if self.plot:
            publishImageScatter(im1, corners5)
            publishImageScatter(im2, corners6)
            publishImageScatter(im1, corners1, corners3, corners5)
            publishImageScatter(im2, corners2, corners4, corners6)

        # Estimating alignment using RANSAC...
        offset_x, offset_y, corners7, corners8 = self.estimateRANSAC(im1, im2, corners5, corners6)
        if self.plot:
            publishImageScatter(im1, corners7)
            publishImageScatter(im2, corners8)
            publishImageScatter(im1, corners1, corners3, corners5, corners7)
            publishImageScatter(im2, corners2, corners4, corners6, corners8)

        correspondences = np.array([corners7.T, corners8.T])
        self.plot = False
        return offset_x, offset_y, correspondences

    def getFeaturesAndCombine(self, im1, im2):
        # x goes up to down, y goes left to right
        offset_x, offset_y, correspondences = self.getAutoCorrespondences(im1, im2)
        panorama = []
        if offset_x >= 0 and offset_y >= 0:
            dim_x = np.max((im1.shape[0], im2.shape[0] + offset_x))
            dim_y = np.max((im1.shape[1], im2.shape[1] + offset_y))
            panorama = np.zeros((dim_x, dim_y, im1.shape[2]))
            panorama[:im1.shape[0], :im1.shape[1]] = im1
            panorama[offset_x:offset_x+im2.shape[0], offset_y:offset_y+im2.shape[1]] = im2
        elif offset_x >= 0 and offset_y < 0:
            dim_x = np.max((im1.shape[0], im2.shape[0] + offset_x))
            dim_y = np.max((im1.shape[1] - offset_y, im2.shape[1]))
            panorama = np.zeros((dim_x, dim_y, im1.shape[2]))
            panorama[:im1.shape[0], -offset_y:im1.shape[1]-offset_y] = im1
            panorama[offset_x:offset_x+im2.shape[0], :im2.shape[1]] = im2
        elif offset_x < 0 and offset_y >= 0:
            dim_x = np.max((im1.shape[0] - offset_x, im2.shape[0]))
            dim_y = np.max((im1.shape[1], im2.shape[1] + offset_y))
            panorama = np.zeros((dim_x, dim_y, im1.shape[2]))
            panorama[-offset_x:im1.shape[0]-offset_x, :im1.shape[1]] = im1
            panorama[:im2.shape[0], offset_y:offset_y+im2.shape[1]] = im2
        elif offset_x < 0 and offset_y < 0:
            dim_x = np.max((im1.shape[0] - offset_x, im2.shape[0]))
            dim_y = np.max((im1.shape[1] - offset_y, im2.shape[1]))
            panorama = np.zeros((dim_x, dim_y, im1.shape[2]))
            panorama[-offset_x:im1.shape[0]-offset_x, -offset_y:im1.shape[1]-offset_y] = im1
            panorama[:im2.shape[0], :im2.shape[1]] = im2
        return offset_x, offset_y, panorama

    def cropPanoramaToWrap(self, img1, img2, offset2):
        _, offset_y, _ = self.getAutoCorrespondences(img2, img1)
        return offset2 + offset_y
        # publishImage(panorama2)
        # panorama3 = np.copy(panorama2)
        # halfway = int(panorama2.shape[1]/2)
        # panorama3[:,:halfway] = panorama2[:,-halfway:]
        # panorama3[:,halfway:] = panorama2[:,:-halfway]
        # publishImage(panorama3)
        # return panorama3




# Main
if __name__ == "__main__":
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    Panorama().runMain()
    print ""
    ### Code to test Poisson(): ###
    #print "Poisson now: "+time.ctime()  # TODO: Remove after tested
    #dest = readimage("data/vlsb/IMG_1892.JPG")
    #source = readimage("data/vlsb/IMG_1892.JPG")
    #panorama = Panorama()
    #dest = panorama.warpImage(dest, 1167, panorama.coneMappingIndices)
    #source = panorama.warpImage(source, 1167, panorama.cylindricalMappingIndices)
    #displayImage(dest)
    #displayImage(source)
    #dest = img_as_float(misc.imresize(dest, 0.25))
    #source = img_as_float(misc.imresize(source, 0.125))
    #mask = np.ones((source.shape[0], source.shape[1]))
    #poisson = PoissonSolver()
    #for y in range(mask.shape[0]):
    #    for x in range(mask.shape[1]):
    #        if source[y, x, 0] == 0 and source[y, x, 1] == 0 and source[y, x, 2] == 0:
    #            mask[y, x] = 0
    #output_img = poisson.poisson(source[:,:,0], dest[:,:,0], mask, (50, 150), poisson.mixed_gradient)
    #output_img = panorama.poisson_pyramid(source, dest, 50, 150)
    #displayImage(output_img)
    #print "Done poisson: "+time.ctime()  # TODO: Remove after tested
    ##C Code to test pyramid blending
    #source = readimage("data/o-brien.jpg")[:,:,0]
    #dest = readimage("data/fishingscene.jpeg")[:,:,0]
    #output = Panorama().pyramid_blend(source, dest, 0, dest.shape[1]-int(source.shape[1]/2), levels=100)
    #publishImage(output)

