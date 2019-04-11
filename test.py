
from __future__ import print_function
import tensorflow as tf
import pickle
import cv2
import h5py
import numpy as np
import sys
import os
import time
import subprocess as sp
import tqdm
batch_size = 1 
epochs = 1000


def tf_build_model(weights_name, params, input_tensor, output_tensor):

    def res_block(x, filters, scope):
        with tf.variable_scope(scope):
            fan_in = int(x.shape[-1])
            tensor_1 = tf.keras.layers.Conv2D(filters, (3,3), padding='same', activation=lambda x:tf.nn.leaky_relu(x, alpha=0.02))(x)
            tensor_2 = tf.keras.layers.Conv2D(fan_in, (3,3), padding='same', activation=None)(x)
            tensor_3 = tf.add_n([x, tensor_2])
            tensor = tf.nn.leaky_relu(tensor_3, alpha=0.02)
            return tensor
    layer_1 = tf.keras.layers.Conv2D(64, (5,5), padding='same', strides=(2,2), activation=lambda x:tf.nn.leaky_relu(x, alpha=0.02))
    tensor = layer_1(input_tensor)
    l1_tensor = tensor
    tensor = res_block(tensor, 128, 'res_1')
    layer_2 = tf.keras.layers.Conv2D(128, (5,5), padding='same', strides=(2,2), activation=lambda x:tf.nn.leaky_relu(x, alpha=0.02))
    tensor = layer_2(tensor)
    l2_tensor = tensor
    tensor = res_block(tensor, 256, 'res_2')
    layer_3 = tf.keras.layers.Conv2D(256, (5,5), padding='same', strides=(2,2), activation=lambda x:tf.nn.leaky_relu(x, alpha=0.02))
    tensor = layer_3(tensor)
    tensor = res_block(tensor, 256, 'res_3')
    tensor = res_block(tensor, 256, 'res_4')
    tensor = res_block(tensor, 256, 'res_5')

    def up_op(x):
        tensor = tf.keras.layers.UpSampling2D()(x)
        return tensor
    tensor = up_op(tensor)
    layer_4 = tf.keras.layers.Conv2D(128, (5, 5), padding='same', strides=(1,1), activation=lambda x:tf.nn.leaky_relu(x, alpha=0.02))
    tensor = layer_4(tensor)
    tensor = res_block(tensor, 256, 'res_6')
    tensor = tf.add_n([tensor, l2_tensor])
    tensor = up_op(tensor)
    layer_5 = tf.keras.layers.Conv2D(64, (5, 5), padding='same', strides=(1,1), activation=lambda x:tf.nn.leaky_relu(x, alpha=0.02))
    tensor = layer_5(tensor)
    tensor = tf.add_n([tensor, l1_tensor])
    tensor = up_op(tensor)
    layer_6 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', strides=(1,1), activation=lambda x:tf.nn.leaky_relu(x, alpha=0.02))
    tensor = layer_6(tensor)
    layer_7 = tf.keras.layers.Conv2D(1, (1, 1), padding='same', strides=(1,1), activation=None)
    tensor = layer_7(tensor)
    return tensor


def evaluate(output_file_name):
    global batch_size
    weights_name = None
    init_lr = 0.001
    
    weights_name = sys.argv[1]

    with open(sys.argv[3],'rb') as f:
        data = pickle.load(f)
    config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)
    config.gpu_options.allow_growth = True

    print("Loading data")
    ress = []
    for name in tqdm.tqdm(data):
        img1, img2 = data[name]
        s = img1.shape
        px = 8 - s[0]%8
        py = 8 - s[1]%8
        img1 = np.pad(img1, ((0, px), (0, py)), 'reflect')
        img2 = np.pad(img2, ((0, px), (0, py)), 'reflect')
        img1 = np.expand_dims(img1, -1).astype(np.float32)/255.
        img2 = np.expand_dims(img2, -1).astype(np.float32)/255.
        tf.reset_default_graph()
        res = tf_build_model(weights_name, {'learning_rate': init_lr}, tf.expand_dims(tf.concat([img1, img2], -1), 0), None)
        saver = tf.train.Saver(max_to_keep=50)
        checkpoint_dir = './ckpt/'
        with tf.Session(config=config) as sess:
            saver.restore(sess, weights_name)
            print('restoring %s' % (weights_name))
            ress.append(sess.run(res))
    with open(output_file_name, 'wb') as f:
        pickle.dump(ress, f)

import sys

if __name__ == '__main__':
    evaluate(sys.argv[2])

