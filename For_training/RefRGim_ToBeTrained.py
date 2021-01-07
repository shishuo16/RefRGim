'''
#1 sample x SNP matrix
#2 sample population matrix: sample x population
#3 output prefix
'''

import tensorflow as tf
import numpy as np
import sys
import math
import os

x_data = np.loadtxt(sys.argv[1])
if x_data.ndim == 1:
    x_data=np.array([x_data])

snp_num = x_data.shape[1]
y_data = np.loadtxt(sys.argv[2])

def next_batch(num, data, labels):
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    idi = idx[:num]
    train_data = [data[i] for i in idi]
    train_labels = [labels[i] for i in idi]
    retD = list(set(idx)-set(idi))
    np.random.shuffle(retD)
    ido = retD[:num]
    val_data = [data[i] for i in ido]
    val_labels = [labels[i] for i in ido]
    return np.asarray(train_data), np.asarray(train_labels),np.asarray(val_data), np.asarray(val_labels)

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1, phase_train: False})
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1, phase_train: False})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def batch_norm(x, n_out, phase_train):
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,mean_var_with_update,lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def add_layer(inputs, in_size_h, in_size_w, in_size_c, out_size, layer_name, activation_function, phase_train, ks_num):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = weight_variable([in_size_h, in_size_w, in_size_c, out_size])
            tf.summary.histogram(".", Weights)
        with tf.name_scope('biases'):
            biases = bias_variable([out_size])
            tf.summary.histogram(".", biases)
        with tf.name_scope('conv2d'):
            wx_plus_b = conv2d(inputs, Weights) +  biases
        with tf.name_scope('bn'):
            wx_plus_b_bn = batch_norm(wx_plus_b, out_size, phase_train)
        with tf.name_scope('activation'):
            act = activation_function(wx_plus_b_bn)
            tf.summary.histogram('.', act)
        with tf.name_scope('max_pooling'):
            results = tf.nn.max_pool(act, ksize=[1,1,ks_num,1], strides=[1,1,ks_num,1], padding='VALID')
    return results

def Spp_layer(feature_map, bins):
    batch_size, _, a, b = feature_map.get_shape().as_list()
    pooling_out_all = list(range(len(bins)))
    for layer in range(len(bins)):
        stride = math.floor(a / bins[layer])
        k_size = (a - stride*bins[layer])+1
        pooling_out = tf.nn.max_pool(feature_map,ksize=[1, 1, k_size, 1],strides=[1, 1, stride, 1],padding='VALID')
        pooling_out_resized = tf.reshape(pooling_out, [-1, ((a-k_size)//stride+1), b])
        pooling_out_all[layer] = pooling_out_resized

    feature_map_out = tf.concat(axis=1, values=pooling_out_all)
    return feature_map_out

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, snp_num],name='x') 
    ys = tf.placeholder(tf.float32, [None, 26],name='labels')

keep_prob = tf.placeholder(tf.float32)
phase_train = tf.placeholder(tf.bool)

with tf.name_scope('input_reshape'):
    x_image = tf.reshape(xs, [-1, 1, snp_num, 1])

## conv layer 1
layer_1 = add_layer(x_image, 1, 15, 1, 32, 'conv1', tf.nn.relu, phase_train, 4)

## conv layer 2
layer_2 = add_layer(layer_1, 1, 15, 32, 64, 'conv2', tf.nn.relu, phase_train, 4)

## conv layer 3
layer_3 = add_layer(layer_2, 1, 15, 64, 128, 'conv3', tf.nn.relu, phase_train, 4)

## conv layer 4
layer_4 = add_layer(layer_3, 1, 15, 128, 128, 'conv4', tf.nn.relu, phase_train, 1)

## conv layer 5
layer_5 = add_layer(layer_4, 1, 15, 128, 128, 'conv5', tf.nn.relu, phase_train, 1)

layer_spp = Spp_layer(layer_5, [6, 4, 2, 1])

with tf.name_scope('fc1'):
    ## fc1 layer 
    with tf.name_scope('weights'):
        W_fc1 = weight_variable([13*128, 512])
    with tf.name_scope('biases'):
        b_fc1 = bias_variable([512])
    with tf.name_scope('flat'):
        h_pool2_flat = tf.reshape(layer_spp, [-1, 13*128])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    with tf.name_scope('dropout'):
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope('fc2'):
    ## fc2 layer 
    with tf.name_scope('weights'):
        W_fc2 = weight_variable([512, 26]) 
    with tf.name_scope('biases'):
        b_fc2 = bias_variable([26])
    with tf.name_scope('prediction'):
        prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

with tf.name_scope('cross_entropy'):
    with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction + 1e-10), reduction_indices=[1]))       
tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)


    

sess = tf.Session()

path1 = sys.argv[3] + "_training_info/"
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(path1, sess.graph)

init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver()
for i in range(5000000):
    batch_xs, batch_ys, test_xs, test_ys = next_batch(200, x_data, y_data)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5, phase_train: True})
    if i % 100 == 0:
        train_acc = compute_accuracy(batch_xs, batch_ys)
        print("--- Training Round:", i, "---")
        print("Training accuracy:", train_acc)
        test_acc = compute_accuracy(test_xs, test_ys)
        print("Testing accuracy:",test_acc)
        result = sess.run(merged, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5, phase_train: True})
        writer.add_summary(result, i)
        if test_acc > 0.999 and train_acc > 0.999:
            all_acc = compute_accuracy(x_data, y_data)
            if all_acc > 0.999:
                break

output = sess.run(prediction, feed_dict={xs: x_data, keep_prob: 1, phase_train: False})
output_path = sys.argv[3] + ".population.probability.txt"
np.savetxt(output_path, output,delimiter='\t')
path2 = sys.argv[3] + "_net/save_net.ckpt"
save_path = saver.save(sess, path2)
print("Save to path: ", save_path)
