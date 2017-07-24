from __future__ import division
import tensorflow as tf
import os, re, time
import numpy as np

from utils import *

def conv3d(input_, output_dim, f_size, is_training, scope='conv3d'):
    with tf.variable_scope(scope) as scope:
        # VGG network uses two 3*3 conv layers to effectively increase receptive field
        w1 = tf.get_variable('w1', [f_size, f_size, f_size, input_.get_shape()[-1], output_dim],
                             initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1 = tf.nn.conv3d(input_, w1, strides=[1, 1, 1, 1, 1], padding='SAME')
        b1 = tf.get_variable('b1', [output_dim], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.bias_add(conv1, b1)
        bn1 = tf.contrib.layers.batch_norm(conv1, is_training=is_training, scope='bn1',
                                           variables_collections=['bn_collections'])
        r1 = tf.nn.relu(bn1)
        
        w2 = tf.get_variable('w2', [f_size, f_size, f_size, output_dim, output_dim],
                             initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2 = tf.nn.conv3d(r1, w2, strides=[1, 1, 1, 1, 1], padding='SAME')
        b2 = tf.get_variable('b2', [output_dim], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.bias_add(conv2, b2)
        bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_training, scope='bn2',
                                           variables_collections=['bn_collections'])
        r2 = tf.nn.relu(bn2)
        return r2
    
def deconv3d(input_, output_shape, f_size, is_training, scope='deconv3d'):
    with tf.variable_scope(scope) as scope:
        output_dim = output_shape[-1]
        w = tf.get_variable('w', [f_size, f_size, f_size, output_dim, input_.get_shape()[-1]],
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
        deconv = tf.nn.conv3d_transpose(input_, w, output_shape, strides=[1, f_size, f_size, f_size, 1], padding='SAME')
        bn = tf.contrib.layers.batch_norm(deconv, is_training=is_training, scope='bn',
                                          variables_collections=['bn_collections'])
        r = tf.nn.relu(bn)
        return r
    
def crop_and_concat(x1, x2):
    x1_shape = x1.get_shape().as_list()
    x2_shape = x2.get_shape().as_list()
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, (x1_shape[3] - x2_shape[3]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], x2_shape[3], -1]
    x1_crop = tf.slice(x1, offsets, size)
    return tf.concat([x1_crop, x2], 4)


# Some code from https://github.com/shiba24/3d-unet.git
class UNet3D(object):
    def __init__(self, sess, checkpoint_dir, log_dir, training_paths, testing_paths, roi, im_size, nclass,
                 batch_size=1, layers=3, features_root=32, conv_size=3, dropout=0.5, testing_gt_available=True,
                 loss_type='cross_entropy', class_weights=None):
        self.sess = sess
        
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        
        self.training_paths = training_paths
        self.testing_paths = testing_paths
        self.testing_gt_available = testing_gt_available
        
        self.nclass = nclass
        self.im_size = im_size
        self.roi = roi # (roi_order, roi_name)
        
        self.batch_size = batch_size
        self.layers = layers
        self.features_root = features_root
        self.conv_size = conv_size
        self.dropout = dropout
        self.loss_type = loss_type
        
        self.class_weights = class_weights
        
        self.build_model()
        
        self.saver = tf.train.Saver(tf.trainable_variables() + tf.get_collection_ref('bn_collections'))
        
    def build_model(self):
        self.images = tf.placeholder(tf.float32, shape=[None, self.im_size[0], self.im_size[1], self.im_size[2],
                                                        1], name='images') # Only support single channel
        self.labels = tf.placeholder(tf.float32, shape=[None, self.im_size[0], self.im_size[1], self.im_size[2],
                                                        self.nclass], name='labels')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.keep_prob = tf.placeholder(tf.float32, name='dropout_ratio')
        
        with tf.variable_scope('constants') as scope:
            self.mean = tf.get_variable('mean', [1])
            self.std = tf.get_variable('std', [1])
        
        conv_size = self.conv_size
        layers = self.layers

        deconv_size = 2
        pool_stride_size = 2
        pool_kernel_size = 3 # Use a larger kernel
        
        # Encoding path
        connection_outputs = []
        for layer in range(layers):
            features = 2**layer * self.features_root
            if layer == 0:
                prev = self.images
            else:
                prev = pool
                
            conv = conv3d(prev, features, conv_size, is_training=self.is_training, scope='encoding' + str(layer))
            connection_outputs.append(conv)
            pool = tf.nn.max_pool3d(conv, ksize=[1, pool_kernel_size, pool_kernel_size, pool_kernel_size, 1],
                                    strides=[1, pool_stride_size, pool_stride_size, pool_stride_size, 1],
                                    padding='SAME')
        
        bottom = conv3d(pool, 2**layers * self.features_root, conv_size, is_training=self.is_training, scope='bottom')
        bottom = tf.nn.dropout(bottom, self.keep_prob)
        
        # Decoding path
        for layer in range(layers):
            conterpart_layer = layers - 1 - layer
            features = 2**conterpart_layer * self.features_root
            if layer == 0:
                prev = bottom
            else:
                prev = conv_decoding
            
            shape = prev.get_shape().as_list()
            deconv_output_shape = [tf.shape(prev)[0], shape[1] * deconv_size, shape[2] * deconv_size,
                                   shape[3] * deconv_size, features]
            deconv = deconv3d(prev, deconv_output_shape, deconv_size, is_training=self.is_training,
                              scope='decoding' + str(conterpart_layer))
            cc = crop_and_concat(connection_outputs[conterpart_layer], deconv)
            conv_decoding = conv3d(cc, features, conv_size, is_training=self.is_training,
                                   scope='decoding' + str(conterpart_layer))
        
        with tf.variable_scope('logits') as scope:
            w = tf.get_variable('w', [1, 1, 1, conv_decoding.get_shape()[-1], self.nclass],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
            logits = tf.nn.conv3d(conv_decoding, w, strides=[1, 1, 1, 1, 1], padding='SAME')
            b = tf.get_variable('b', [self.nclass], initializer=tf.constant_initializer(0.0))
            logits = tf.nn.bias_add(logits, b)
        
        self.probs = tf.nn.softmax(logits)
        self.predictions = tf.argmax(self.probs, 4)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predictions, tf.argmax(self.labels, 4)), tf.float32))
                                  
        flat_logits = tf.reshape(logits, [-1, self.nclass])
        flat_labels = tf.reshape(self.labels, [-1, self.nclass])
        
        if self.class_weights is not None:
            class_weights = tf.constant(np.asarray(self.class_weights, dtype=np.float32))
            weight_map = tf.reduce_max(tf.multiply(flat_labels, class_weights), axis=1)
            loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, labels=flat_labels)
            weighted_loss = tf.multiply(loss_map, weight_map)
            cross_entropy_loss = tf.reduce_mean(weighted_loss)
        else:
            cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
                                                                                        labels=flat_labels))
        eps = 1e-5
        dice_value = 0
        dice_loss = 0
        for i in range(1, self.nclass):
            slice_prob = tf.squeeze(tf.slice(self.probs, [0, 0, 0, 0, i], [-1, -1, -1, -1, 1]), axis=4)
            slice_prediction = tf.cast(tf.equal(self.predictions, i), tf.float32)
            slice_label = tf.squeeze(tf.slice(self.labels, [0, 0, 0, 0, i], [-1, -1, -1, -1, 1]), axis=4)
            intersection_prob = tf.reduce_sum(tf.multiply(slice_prob, slice_label), axis=[1, 2, 3])
            intersection_prediction = tf.reduce_sum(tf.multiply(slice_prediction, slice_label), axis=[1, 2, 3])
            union = eps + tf.reduce_sum(slice_prediction, axis=[1, 2, 3]) + tf.reduce_sum(slice_label, axis=[1, 2, 3])
            dice_loss += tf.reduce_mean(tf.div(intersection_prob, union))
            dice_value += tf.reduce_mean(tf.div(intersection_prediction, union))
        dice_value = dice_value * 2.0 / (self.nclass - 1)
        dice_loss = 1 - dice_loss * 2.0 / (self.nclass - 1)
        self.dice = dice_value
        
        if self.loss_type == 'cross_entropy':
            self.loss = cross_entropy_loss
        elif self.loss_type == 'dice':
            self.loss = cross_entropy_loss + dice_loss
        else:
            raise ValueError("Unknown cost function: " + self.loss_type)
        
        self.loss_summary = tf.summary.scalar('loss', self.loss)
        self.accuracy_summary = tf.summary.scalar('accuracy', self.accuracy)
        self.dice_summary = tf.summary.scalar('dice', self.dice)
    
    def train(self, config):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer().minimize(self.loss)
        
        self.sess.run(tf.global_variables_initializer())
        
        train_writer = tf.summary.FileWriter(os.path.join(self.log_dir, self.model_dir, 'train'), self.sess.graph)
        if self.testing_gt_available:
            test_writer = tf.summary.FileWriter(os.path.join(self.log_dir, self.model_dir, 'test'))
        
        merged = tf.summary.merge([self.loss_summary, self.accuracy_summary, self.dice_summary])
        
        counter = 0
        
        for epoch in range(config['epoch']):
            training_paths = np.random.permutation(self.training_paths)
            if epoch == 0:
                mean, std = self.estimate_mean_std()
                self.sess.run([self.mean.assign([mean]), self.std.assign([std])])
                
            for f in range(len(training_paths) // self.batch_size):
                images = np.empty((self.batch_size, self.im_size[0], self.im_size[1], self.im_size[2], 1),
                                  dtype=np.float32)
                labels = np.empty((self.batch_size, self.im_size[0], self.im_size[1], self.im_size[2], self.nclass),
                                  dtype=np.float32)
                for b in range(self.batch_size):
                    order = f * self.batch_size + b
                    images[b, ..., 0], labels[b] = read_training_inputs(training_paths[order], self.roi[0], self.im_size)
                    
                images = (images - mean) / std
                _, train_loss, summary = self.sess.run([optimizer, self.loss, merged],
                                                       feed_dict={ self.images: images,
                                                                   self.labels: labels,
                                                                   self.is_training: True,
                                                                   self.keep_prob: self.dropout })
                train_writer.add_summary(summary, counter)
                counter += 1
                if np.mod(counter, 1000) == 0:
                    self.save(counter)
                
                # Test during training, for simplicity, this relies on the label information to extract images so that it
                # only tells whether the trained model can segment the roi well given the extracted images
                if self.testing_gt_available and np.mod(counter, 100) == 0:
                    testing_paths = np.random.permutation(self.testing_paths)
                    for b in range(self.batch_size):
                        images[b, ..., 0], labels[b] = read_training_inputs(testing_paths[b], self.roi[0], self.im_size)
                    images = (images - mean) / std
                    test_loss, summary = self.sess.run([self.loss, merged],
                                                       feed_dict = { self.images: images,
                                                                     self.labels: labels,
                                                                     self.is_training: True,
                                                                     self.keep_prob: 1 })
                    test_writer.add_summary(summary, counter)
                
                    print(self.roi[1] + "_" + str(counter) + ":" + "train_loss: " + \
                          str(train_loss) + " test_loss: " + str(test_loss))
                    
        # Save in the end
        self.save(counter)
    
    def test(self, input_path, output_path):
        if not self.load()[0]:
            raise Exception("No model is found, please train first")
            
        mean, std = self.sess.run([self.mean, self.std])
        
        images = np.empty((1, self.im_size[0], self.im_size[1], self.im_size[2], 1), dtype=np.float32)
        #labels = np.empty((1, self.im_size[0], self.im_size[1], self.im_size[2], self.nclass), dtype=np.float32)
        for f in input_path:
            images[0, ..., 0], read_info = read_testing_inputs(f, self.roi[0], self.im_size, output_path)
            probs = self.sess.run(self.probs, feed_dict = { self.images: (images - mean) / std,
                                                            self.is_training: True,
                                                            self.keep_prob: 1 })
            #print(self.roi[1] + os.path.basename(f) + ":" + str(dice))
            output_file = os.path.join(output_path, self.roi[1] + '_' + os.path.basename(f))
            f_h5 = h5py.File(output_file, 'w')
            if self.roi[0] < 0:
                f_h5['predictions'] = restore_labels(np.argmax(probs[0], 3), self.roi[0], read_info)
            else:
                f_h5['probs'] = restore_labels(probs[0, ..., 1], self.roi[0], read_info)
            f_h5.close()
    
    def estimate_mean_std(self):
        means = []
        stds = []
        # Strictly speaking, this is not the correct way to estimate std since the mean 
        # used in each image is not the global mean but the mean of the image, this would
        # cause an over-estimation of the std.
        # The correct way may need much more memory, and more importantly, it probably does not matter...
        for i in range(100):
            n = np.random.choice(len(self.training_paths))
            images, _ = read_training_inputs(self.training_paths[n], self.roi[0], self.im_size)
            means.append(np.mean(images))
            stds.append(np.std(images))
        return np.mean(means), np.mean(stds)
    
    @property
    def model_dir(self):
        return "{}_unet3d_layer{}_{}".format(self.roi[1], self.layers, self.loss_type)
    
    def save(self, step, model_name='main'):
        checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
        
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)
        
    def load(self, model_name='main'):
        checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
        
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            return True, counter
        else:
            print("Failed to find a checkpoint")
            return False, 0