# Data augmented
from __future__ import print_function
import tensorflow as tf
import cv2
import h5py
import numpy as np
import sys
import os
import time
import subprocess as sp

batch_size = 64 
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

    train_mse = tf.reduce_mean(tf.squared_difference(output_tensor, tensor))

    global_step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(loss=train_mse, global_step=tf.train.get_global_step())

    return train_op, train_mse

def drive():
    global batch_size
    weights_name = None
    init_lr = 0.0001
    
    print(weights_name)
    
    hf = h5py.File('nova_data_v4.h5', 'r')

    config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)
    config.gpu_options.allow_growth = True
    # with tf.Session(config=config) as sess:
    #     pass
    print("Loading data")

    x = np.array(hf['data'])
    y = np.array(hf['label'])
    length = x.shape[0]
    shuffle_list = np.arange(length)
    bar = int(length * 0.9)
    bar = bar - bar % batch_size
    np.random.shuffle(shuffle_list)
    dx = x[shuffle_list][:bar]
    dy = y[shuffle_list][:bar]
    lx = x[shuffle_list][bar:]
    ly = y[shuffle_list][bar:]

    num_scale = 128

    def train_generator():
        x = np.zeros([batch_size, num_scale, num_scale, 2], dtype=np.float32)
        y = np.zeros([batch_size, num_scale, num_scale, 1], dtype=np.float32)
        while True:
            for i in range(0, bar, batch_size)[:-1]:
                x[:,:,:,:] = dx[i:i+batch_size,...]
                y[:,:,:,:] = dy[i:i+batch_size,...]
                yield x/255., y/255.


    def val_generator():
        x = np.zeros([batch_size, num_scale, num_scale, 2], dtype=np.float32)
        y = np.zeros([batch_size, num_scale, num_scale, 1], dtype=np.float32)
        for i in range(0, length-bar, batch_size)[:-1]:
            x[:,:,:,:] = lx[i:i+batch_size,...]
            y[:,:,:,:] = ly[i:i+batch_size,...]
            yield x/255., y/255.

    inputs = tf.placeholder(tf.float32, [batch_size, num_scale, num_scale, 2])
    targets = tf.placeholder(tf.float32, [batch_size, num_scale, num_scale, 1])

    # build model
    train_op, mse_loss = tf_build_model(
                                       weights_name,
                                       {'learning_rate': init_lr},
                                       inputs,
                                       targets)
    
    saver = tf.train.Saver(max_to_keep=50)
    checkpoint_dir = './ckpt/'
    with tf.Session(config=config) as sess:
        if weights_name is not None:
            saver.restore(sess, weights_name)
        else:
            sess.run(tf.global_variables_initializer())
        total_var = 0
        options = tf.RunOptions()  # trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        data_gen = train_generator()
        interval = 500
        metrics = np.zeros((interval,2))
        train_tot_step = int(sys.argv[1])
        for i in range(train_tot_step):
            if i % interval == 0:
                val_mse_s = []
                val_gen = val_generator()
                for v_data, v_label in val_gen:
                    val_mse = sess.run(mse_loss, feed_dict={
                                                 inputs: v_data, targets: v_label})
                    val_mse_s.append(float(val_mse))
                # print(val_satd_s)
                print("[%s] step %8d" % (time.asctime( time.localtime() ), i))
                print("Train MSE %.8f, Val MSE %.8f" % (
                    np.mean(metrics[:,0]), np.mean(val_mse_s)))
                
            iter_data, iter_label = next(data_gen)
            # print(iter_data.shape)
            feed_dict = {inputs: iter_data, targets: iter_label}
            _, mse = sess.run([train_op, mse_loss],
                                   feed_dict=feed_dict,
                                    options=options,
                                    run_metadata=run_metadata)
            metrics[i%interval,0] = mse
            
            if i % interval == 0:
                save_path = saver.save(sess, os.path.join(
                    checkpoint_dir, "model_%06d.ckpt" % (i)))

if __name__ == '__main__':
    drive()
