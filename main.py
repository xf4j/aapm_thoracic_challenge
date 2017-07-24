from __future__ import division
import os
import numpy as np
import pickle
import pprint
import tensorflow as tf

from utils import *
from constants import *
from model import UNet3D

flags = tf.app.flags
flags.DEFINE_integer("epoch", 200, "Epoch to train [200]")
flags.DEFINE_string("train_data_dir", "data_training", "Directory name of the data [data_training]")
flags.DEFINE_string("test_data_dir", "data_testing", "Directory name of the test data [data_testing]")
flags.DEFINE_string("output_dir", "data_output", "Directory name of the output data [data_output]")
flags.DEFINE_integer("step1_features_root", 24, "Number of features in the first filter in step 1 [24]")
flags.DEFINE_integer("step2_features_root", 48, "Number of features in the first filter [48]")
flags.DEFINE_integer("conv_size", 3, "Convolution kernel size in encoding and decoding paths [3]")
flags.DEFINE_integer("layers", 3, "Encoding and deconding layers [3]")
flags.DEFINE_string("loss_type", "cross_entropy", "Loss type in the model [cross_entropy]")
flags.DEFINE_float("dropout_ratio", 0.5, "Drop out ratio [0.5]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("log_dir", "logs", "Directory name to save logs [logs]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [True]")
FLAGS = flags.FLAGS

def main(_):
    pp = pprint.PrettyPrinter()
    pp.pprint(flags.FLAGS.__flags)
    
    if FLAGS.test_data_dir == FLAGS.train_data_dir:
        testing_gt_available = True
        if os.path.exists(os.path.join(FLAGS.train_data_dir, 'files.log')):
            with open(os.path.join(FLAGS.train_data_dir, 'files.log'), 'r') as f:
                training_paths, testing_paths = pickle.load(f)
        else:
            # Phase 0
            all_subjects = [os.path.join(FLAGS.train_data_dir, name) for name in os.listdir(FLAGS.train_data_dir)]
            n_training = int(np.rint(len(all_subjects) * 2 / 3))
            training_paths = all_subjects[:n_training]
            testing_paths = all_subjects[n_training:]
            # Save the training paths and testing paths
            with open(os.path.join(FLAGS.train_data_dir, 'files.log'), 'w') as f:
                pickle.dump([training_paths, testing_paths], f)
    else:
        testing_gt_available = False
        training_paths = [os.path.join(FLAGS.train_data_dir, name)
                          for name in os.listdir(FLAGS.train_data_dir) if '.hdf5' in name]
        testing_paths = [os.path.join(FLAGS.test_data_dir, name)
                         for name in os.listdir(FLAGS.test_data_dir) if '.hdf5' in name]
        
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    
    run_config = tf.ConfigProto()
    with tf.Session(config=run_config) as sess:
        unet_all = UNet3D(sess, checkpoint_dir=FLAGS.checkpoint_dir, log_dir=FLAGS.log_dir, training_paths=training_paths,
                          testing_paths=testing_paths, nclass=N_CLASSES + 1, layers=FLAGS.layers,
                          features_root=FLAGS.step1_features_root, conv_size=FLAGS.conv_size, dropout=FLAGS.dropout_ratio,
                          loss_type=FLAGS.loss_type, roi=(-1, 'All'), im_size=ALL_IM_SIZE,
                          testing_gt_available=testing_gt_available, class_weights=(1.0, 2.0, 1.0, 1.0, 1.0, 3.0))
        if FLAGS.train:
            train_config = {}
            train_config['epoch'] = FLAGS.epoch
            unet_all.train(train_config)
        else:
            if not os.path.exists(FLAGS.output_dir):
                os.makedirs(FLAGS.output_dir)
                    
            unet_all.test(testing_paths, FLAGS.output_dir)

    tf.reset_default_graph()
    
    # Second step training
    rois = ['SpinalCord', 'Lung_R', 'Lung_L', 'Heart', 'Esophagus']
    im_sizes = [(160, 128, 64), (72, 192, 120), (72, 192, 120), (32, 160, 192), (80, 80, 64)]
    weights = [(1.0, 2.0), (1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (1.0, 3.0)]
        
    for roi in range(5):
        run_config = tf.ConfigProto()
        # Build model
        with tf.Session(config=run_config) as sess:
            unet = UNet3D(sess, checkpoint_dir=FLAGS.checkpoint_dir, log_dir=FLAGS.log_dir, training_paths=training_paths,
                          testing_paths=testing_paths, nclass=2, layers=FLAGS.layers, features_root=FLAGS.step2_features_root,
                          conv_size=FLAGS.conv_size, dropout=FLAGS.dropout_ratio, loss_type=FLAGS.loss_type,
                          roi=(roi, rois[roi]), im_size=im_sizes[roi], testing_gt_available=testing_gt_available,
                          class_weights=weights[roi])
            
            if FLAGS.train:
                train_config = {}
                train_config['epoch'] = FLAGS.epoch
                unet.train(train_config)
            else:
                if not os.path.exists(FLAGS.output_dir):
                    os.makedirs(FLAGS.output_dir)
                    
                # Get result for single ROI
                unet.test(testing_paths, FLAGS.output_dir)
                
        tf.reset_default_graph()
    
        
if __name__ == '__main__':
    tf.app.run()
