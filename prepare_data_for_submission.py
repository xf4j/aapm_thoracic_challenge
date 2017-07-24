from __future__ import division
import os, sys, glob
import numpy as np
import dicom
from skimage.draw import polygon
from skimage.transform import resize
import h5py
import SimpleITK as sitk

from constants import *
from utils import *
    
def read_images_info(path):
    for subdir, dirs, files in os.walk(path):
        dcms = glob.glob(os.path.join(subdir, '*.dcm'))
        if len(dcms) > 1:
            slices = [dicom.read_file(dcm) for dcm in dcms]
            slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
            images = np.stack([s.pixel_array for s in slices], axis=0).astype(np.float32)
            images = images + slices[0].RescaleIntercept
    
    orig_shape = images.shape
    
    inplane_scale = slices[0].PixelSpacing[0] / PIXEL_SPACING
    inplane_size = int(np.rint(inplane_scale * slices[0].Rows / 2) * 2)
    return orig_shape, inplane_size
    
if __name__ == '__main__':
    raw_input_path = '../DOI_Offline_Testing'
    result_path = 'data_output'
    output_path = 'submit'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    rois = ['SpinalCord', 'Lung_R', 'Lung_L', 'Heart', 'Esophagus']
    
    subjects = [os.path.join(raw_input_path, name)
                for name in sorted(os.listdir(raw_input_path)) if os.path.isdir(os.path.join(raw_input_path, name))]

    for sub in subjects:
        print(sub)
        name = os.path.basename(sub)
        orig_shape, inplane_size = read_images_info(sub)
        labels = np.zeros(orig_shape, dtype=np.int16)
        max_probs = np.zeros(orig_shape, dtype=np.float32)
        for nroi in range(len(rois)):
            roi = rois[nroi]
            f = h5py.File(os.path.join(result_path, roi + '_' + name + '.hdf5'), 'r')
            probs = np.asarray(f['probs'], dtype=np.float32)
            f.close()
            if inplane_size < probs.shape[1]:
                crop = int((probs.shape[1] - inplane_size) / 2)
                probs = probs[:, crop : crop + INPLANE_SIZE, crop : crop + INPLANE_SIZE]
            elif inplane_size > probs.shape[1]:
                pad = int((inplane_size - probs.shape[1]) / 2)
                probs = np.pad(probs, ((0, 0), (pad, pad), (pad, pad)), 'constant')
            if orig_shape != probs.shape:
                probs = resize(probs, orig_shape, mode='constant')
            probs[probs < 0.5] = 0 # Ignore those classfied as background
            labels[np.logical_and(probs >= 0.5, probs >= max_probs)] = nroi + 1
            max_probs = np.maximum(probs, max_probs)
        
        reader = sitk.ImageSeriesReader()
        for subdir, dirs, files in os.walk(sub):
            dcms = glob.glob(os.path.join(subdir, '*.dcm'))
            if len(dcms) > 1:
                orig_images = sitk.ReadImage(reader.GetGDCMSeriesFileNames(subdir))
        img = sitk.GetImageFromArray(labels)
        img.CopyInformation(orig_images)
        sitk.WriteImage(img, os.path.join(output_path, name + '_labels.mha'))