## Synopsis

Submission for AAPM Thoracic Auto-segmentation Challenge (http://aapmchallenges.cloudapp.net/competitions/3). A two-step 3D U-Net model is used with the first step segmenting all ROIs to obtain the bounding boxes and the second step taking the bounding boxes to segment foreground/background for each ROI.

## Code Example

The workflow includes pre-processing, training, post-processing, testing and submission.</br><br/>
After training, offline testing and live testing data is downloaded, run `python convert_data.py input_dir output_dir` to finish pre-processing including voxel size and intensity normalization and contour extraction, etc.</br><br/>
To train the model using the training dataset, run `python main.py --train=True --train_data_dir=train_output_dir --test_data_dir=test_output_dir`. Or you can modify the default parameters in `main.py` so that you can just run `python main.py`. Not all model parameters are optimized, such as `epoch`, `step1_features_root`, `step2_features_root`, `conv_size`, `dropout_ratio`, but the current default values give a satisfactory result. Check `model.py` for more details about the network structure.<br/><br/>
To test the model on validation dataset, run `python main.py --train=False --train_data_dir=train_output_dir --test_data_dir=test_output_dir --output_dir=output_dir`. The results will be saved at `output_dir` including outputs from both steps.<br/><br/>
To combine the results and generate the final label maps, run `python prepare_data_for_submission.py` after modifying these values `raw_input_path`, `result_path` and `output_path` in the script, corresponding to the raw data downloaded from the challenge website, the results generated by the model and the output for submission, respectively. `.mha` format is used for the submission. Or you can modify this script to save in other formats.

## Motivation

This project demonstrates that using a two step model can improve the segmentation accuracy for organs in thoracic CT images and provides a way to crop the 3D images for the model to fit into the memory.

## Installation

The model is implemented and tested using `python 2.7` and `Tensorflow 1.1.0`, but `python 3` and newer versions of `Tensorflow` should also work.
Other required libraries include: `numpy`, `h5py`, `skimage`, `transforms3d`, `SimpleITK`.

## Contributors

Xue Feng, Department of Biomedical Engineering, University of Virginia
xf4j@virginia.edu