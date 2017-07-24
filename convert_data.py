from __future__ import division
import os, sys, glob
import numpy as np
import dicom
from skimage.draw import polygon
from skimage.transform import resize
import h5py
from constants import *
from utils import *

def read_structure(structure):
    contours = []
    for i in range(len(structure.ROIContourSequence)):
        contour = {}
        contour['color'] = structure.ROIContourSequence[i].ROIDisplayColor
        contour['number'] = structure.ROIContourSequence[i].RefdROINumber
        contour['name'] = structure.StructureSetROISequence[i].ROIName
        assert contour['number'] == structure.StructureSetROISequence[i].ROINumber
        contour['contours'] = [s.ContourData for s in structure.ROIContourSequence[i].ContourSequence]
        contours.append(contour)
    return contours

def get_labels(contours, shape, slices):
    z = [np.around(s.ImagePositionPatient[2], 1) for s in slices]
    pos_r = slices[0].ImagePositionPatient[1]
    spacing_r = slices[0].PixelSpacing[1]
    pos_c = slices[0].ImagePositionPatient[0]
    spacing_c = slices[0].PixelSpacing[0]
    
    label_map = np.zeros(shape, dtype=np.float32)
    for con in contours:
        num = ROI_ORDER.index(con['name']) + 1
        for c in con['contours']:
            nodes = np.array(c).reshape((-1, 3))
            assert np.amax(np.abs(np.diff(nodes[:, 2]))) == 0
            z_index = z.index(np.around(nodes[0, 2], 1))
            r = (nodes[:, 1] - pos_r) / spacing_r
            c = (nodes[:, 0] - pos_c) / spacing_c
            rr, cc = polygon(r, c)
            label_map[z_index, rr, cc] = num
    
    return label_map
    
def read_images(path):
    for subdir, dirs, files in os.walk(path):
        dcms = glob.glob(os.path.join(subdir, '*.dcm'))
        if len(dcms) > 1:
            slices = [dicom.read_file(dcm) for dcm in dcms]
            slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
            images = np.stack([s.pixel_array for s in slices], axis=0).astype(np.float32)
            images = images + slices[0].RescaleIntercept
    images = normalize(images)
    
    inplane_scale = slices[0].PixelSpacing[0] / PIXEL_SPACING
    inplane_size = int(np.rint(inplane_scale * slices[0].Rows / 2) * 2)
    z_scale = slices[0].SliceThickness / SLICE_THICKNESS
    z_size = int(np.rint(z_scale * images.shape[0]))
    
    if inplane_size != INPLANE_SIZE or z_scale != 1:
        images = resize(images, (z_size, inplane_size, inplane_size), mode='constant')
        if inplane_size != INPLANE_SIZE:
            if inplane_size > INPLANE_SIZE:
                crop = int((inplane_size - INPLANE_SIZE) / 2)
                images = images[:, crop : crop + INPLANE_SIZE, crop : crop + INPLANE_SIZE]
            else:
                pad = int((INPLANE_SIZE - new_size) / 2)
                images = np.pad(images, ((0, 0), (pad, pad), (pad, pad)))
    
    return images
            
def read_images_labels(path):
    # Read the images and labels from a folder containing both dicom files
    for subdir, dirs, files in os.walk(path):
        dcms = glob.glob(os.path.join(subdir, '*.dcm'))
        if len(dcms) == 1:
            structure = dicom.read_file(dcms[0])
            contours = read_structure(structure)
        elif len(dcms) > 1:
            slices = [dicom.read_file(dcm) for dcm in dcms]
            slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
            images = np.stack([s.pixel_array for s in slices], axis=0).astype(np.float32)
            images = images + slices[0].RescaleIntercept
    
    images = normalize(images)
    labels = get_labels(contours, images.shape, slices)
    inplane_scale = slices[0].PixelSpacing[0] / PIXEL_SPACING
    inplane_size = int(np.rint(inplane_scale * slices[0].Rows / 2) * 2)
    z_scale = slices[0].SliceThickness / SLICE_THICKNESS
    z_size = int(np.rint(z_scale * images.shape[0]))
    
    if inplane_size != INPLANE_SIZE or z_scale != 1:
        images = resize(images, (z_size, inplane_size, inplane_size), mode='constant')
        new_labels = np.zeros_like(images, dtype=np.float32)
        for z in range(N_CLASSES):
            roi = resize((labels == z + 1).astype(np.float32), (z_size, inplane_size, inplane_size), mode='constant')
            new_labels[roi >= 0.5] = z + 1
        labels = new_labels
        if inplane_size != INPLANE_SIZE:
            if inplane_size > INPLANE_SIZE:
                crop = int((inplane_size - INPLANE_SIZE) / 2)
                images = images[:, crop : crop + INPLANE_SIZE, crop : crop + INPLANE_SIZE]
                labels = labels[:, crop : crop + INPLANE_SIZE, crop : crop + INPLANE_SIZE]
            else:
                pad = int((INPLANE_SIZE - new_size) / 2)
                images = np.pad(images, ((0, 0), (pad, pad), (pad, pad)), 'constant')
                labels = np.pad(labels, ((0, 0), (pad, pad), (pad, pad)), 'constant')
    
    return images, labels
    
if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise Exception("Need at least the input data directory")
    input_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    else:
        output_path = './data'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    subjects = [os.path.join(input_path, name)
                for name in sorted(os.listdir(input_path)) if os.path.isdir(os.path.join(input_path, name))]

    for sub in subjects:
        name = os.path.basename(sub)
        if 'Training' in input_path:
            images, labels = read_images_labels(sub)
            resized_images, resized_labels = resize_images_labels(images, labels)
            hdf5_file = h5py.File(os.path.join(output_path, name + '.hdf5'), 'w')
            hdf5_file.create_dataset('images', data=images)
            hdf5_file.create_dataset('labels', data=labels)
            hdf5_file.create_dataset('resized_images', data=resized_images)
            hdf5_file.create_dataset('resized_labels', data=resized_labels)
            hdf5_file.close()
        else:
            images = read_images(sub)
            resized_images = resize_images(images)
            hdf5_file = h5py.File(os.path.join(output_path, name + '.hdf5'), 'w')
            hdf5_file.create_dataset('images', data=images)
            hdf5_file.create_dataset('resized_images', data=resized_images)