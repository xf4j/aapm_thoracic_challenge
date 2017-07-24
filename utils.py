from __future__ import division
import os, glob, time
import numpy as np
import h5py
from skimage.transform import resize, warp, AffineTransform
from skimage import measure
from transforms3d.euler import euler2mat
from transforms3d.affines import compose
from constants import *

def normalize(im_input):
    im_output = im_input + 1000 # We want to have air value to be 0 since HU of air is -1000
    # Intensity crop
    im_output[im_output < 0] = 0
    im_output[im_output > 1600] = 1600 # Kind of arbitrary to select the range from -1000 to 600 in HU
    im_output = im_output / 1600.0
    return im_output

def resize_images_labels(images, labels):
    resized_images = resize_images(images)
    # labels
    size = (ALL_IM_SIZE[0], ALL_IM_SIZE[1] + CROP * 2, ALL_IM_SIZE[2] + CROP * 2)
    resized_labels = np.zeros(size, dtype=np.float32)
    for z in range(N_CLASSES):
        roi = resize((labels == z + 1).astype(np.float32), size, mode='constant')
        resized_labels[roi >= 0.5] = z + 1
    resized_labels = resized_labels[:, CROP:-CROP, CROP:-CROP]
    return resized_images, resized_labels

def resize_images(images):
    size = (ALL_IM_SIZE[0], ALL_IM_SIZE[1] + CROP * 2, ALL_IM_SIZE[2] + CROP * 2)
    resized_images = resize(images, size, mode='constant')
    resized_images = resized_images[:, CROP:-CROP, CROP:-CROP]
    return resized_images

def get_tform_coords(im_size):
    coords0, coords1, coords2 = np.mgrid[:im_size[0], :im_size[1], :im_size[2]]
    coords = np.array([coords0 - im_size[0] / 2, coords1 - im_size[1] / 2, coords2 - im_size[2] / 2])
    return np.append(coords.reshape(3, -1), np.ones((1, np.prod(im_size))), axis=0)

def clean_contour(in_contour, is_prob=False):
    if is_prob:
        pred = (in_contour >= 0.5).astype(np.float32)
    else:
        pred = in_contour
    labels = measure.label(pred)
    area = []
    for l in range(1, np.amax(labels) + 1):
        area.append(np.sum(labels == l))
    out_contour = in_contour
    out_contour[np.logical_and(labels > 0, labels != np.argmax(area) + 1)] = 0
    return out_contour

def restore_labels(labels, roi, read_info):
    if roi == -1:
        # Pad first, then resize to original shape
        labels = np.pad(labels, ((0, 0), (CROP, CROP), (CROP, CROP)), 'constant')
        restored_labels = np.zeros(read_info['shape'], dtype=np.float32)
        for z in range(N_CLASSES):
            roi = resize((labels == z + 1).astype(np.float32), read_info['shape'], mode='constant')
            roi[roi >= 0.5] = 1
            roi[roi < 0.5] = 0
            roi = clean_contour(roi, is_prob=False)
            restored_labels[roi == 1] = z + 1
    else:
        labels = clean_contour(labels, is_prob=True)
        # Resize to extracted shape, then pad to original shape
        labels = resize(labels, read_info['extract_shape'], mode='constant')
        restored_labels = np.zeros(read_info['shape'], dtype=np.float32)
        extract = read_info['extract']
        restored_labels[extract[0][0] : extract[0][1], extract[1][0] : extract[1][1], extract[2][0] : extract[2][1]] = labels
    return restored_labels

def read_testing_inputs(file, roi, im_size, output_path=None):
    f_h5 = h5py.File(file, 'r')
    if roi == -1:
        images = np.asarray(f_h5['resized_images'], dtype=np.float32)
        read_info = {}
        read_info['shape'] = np.asarray(f_h5['images'], dtype=np.float32).shape
    else:
        images = np.asarray(f_h5['images'], dtype=np.float32)
        output = h5py.File(os.path.join(output_path, 'All_' + os.path.basename(file)), 'r')
        predictions = np.asarray(output['predictions'], dtype=np.float32)
        output.close()
        # Select the roi
        roi_labels = (predictions == roi + 1).astype(np.float32)
        nz = np.nonzero(roi_labels)
        extract = []
        for c in range(3):
            start = np.amin(nz[c])
            end = np.amax(nz[c])
            r = end - start
            extract.append((np.maximum(int(np.rint(start - r * 0.1)), 0),
                            np.minimum(int(np.rint(end + r * 0.1)), images.shape[c])))

        extract_images = images[extract[0][0] : extract[0][1], extract[1][0] : extract[1][1], extract[2][0] : extract[2][1]]
        read_info = {}
        read_info['shape'] = images.shape
        read_info['extract_shape'] = extract_images.shape
        read_info['extract'] = extract

        images = resize(extract_images, im_size, mode='constant')
    
    f_h5.close()
    return images, read_info
    
def read_training_inputs(file, roi, im_size):
    f_h5 = h5py.File(file, 'r')
    if roi == -1:
        images = np.asarray(f_h5['resized_images'], dtype=np.float32)
        labels = np.asarray(f_h5['resized_labels'], dtype=np.float32)
    else:
        images = np.asarray(f_h5['images'], dtype=np.float32)
        labels = np.asarray(f_h5['labels'], dtype=np.float32)
    f_h5.close()
    
    if roi == -1:
        # Select all
        assert im_size == images.shape
        
        translation = [0, np.random.uniform(-8, 8), np.random.uniform(-8, 8)]
        rotation = euler2mat(np.random.uniform(-5, 5) / 180.0 * np.pi, 0, 0, 'sxyz')
        scale = [1, np.random.uniform(0.9, 1.1), np.random.uniform(0.9, 1.1)]
        warp_mat = compose(translation, rotation, scale)
        tform_coords = get_tform_coords(im_size)
        w = np.dot(warp_mat, tform_coords)
        w[0] = w[0] + im_size[0] / 2
        w[1] = w[1] + im_size[1] / 2
        w[2] = w[2] + im_size[2] / 2
        warp_coords = w[0:3].reshape(3, im_size[0], im_size[1], im_size[2])
        
        final_images = warp(images, warp_coords)
        
        nclass = int(np.amax(labels)) + 1
        final_labels = np.empty(im_size + (nclass,), dtype=np.float32)
        for z in range(1, nclass):
            temp = warp((labels == z).astype(np.float32), warp_coords)
            temp[temp < 0.5] = 0
            temp[temp >= 0.5] = 1
            final_labels[..., z] = temp
        final_labels[..., 0] = np.amax(final_labels[..., 1:], axis=3) == 0   
    else:
        # Select the roi
        roi_labels = (labels == roi + 1).astype(np.float32)

        # Rotate the images and labels
        rotation = np.random.uniform(-15, 15)
        shear = np.random.uniform(-5, 5)
        tf = AffineTransform(rotation=np.deg2rad(rotation), shear=np.deg2rad(shear))
        for z in range(images.shape[0]):
            images[z] = warp(images[z], tf.inverse)
            roi_labels[z] = warp(roi_labels[z], tf.inverse)

        nz = np.nonzero(roi_labels)
        extract = []
        for c in range(3):
            start = np.amin(nz[c])
            end = np.amax(nz[c])
            r = end - start
            extract.append((np.maximum(int(np.rint(start - r * np.random.uniform(0.06, 0.14))), 0),
                            np.minimum(int(np.rint(end + r * np.random.uniform(0.06, 0.14))), images.shape[c])))

        extract_images = images[extract[0][0] : extract[0][1], extract[1][0] : extract[1][1], extract[2][0] : extract[2][1]]
        extract_labels = roi_labels[extract[0][0] : extract[0][1], extract[1][0] : extract[1][1], 
                                    extract[2][0] : extract[2][1]]

        final_images = resize(extract_images, im_size, mode='constant')

        final_labels = np.zeros(im_size + (2,), dtype=np.float32)
        lab = resize(extract_labels, im_size, mode='constant')
        final_labels[lab < 0.5, 0] = 1
        final_labels[lab >= 0.5, 1] = 1
    
    return final_images, final_labels