import tensorflow as tf
import numpy as np
import math
import sys
import re
import os
import bz2

hgimp_path = sys.argv[3]
pos = {}
with bz2.open(hgimp_path + "/1KGP_CNN_net/SNP.INFO.bz2" , 'rt') as f:
    for line in f:
        geno = line.strip('\n').split('\t')
        pos[geno[0]+"_"+geno[1]+"_"+geno[2]+"_"+geno[3]] = geno[0] + "/" + geno[4];

sample_l = []
rev_pos = {}
SNP_n = 0
with open(sys.argv[1]) as f:
    for line in f:
        if re.match('##', line):
            continue
        if re.match('#', line):
            sample = line.strip('\n').split('\t')
            for i in range(9, len(sample)):
                sample_l.append(sample[i])
            continue
        SNP_n = SNP_n + 1
        arr = line.strip('\n').split('\t', 6)[0:5]
        arr[0] = re.sub('chr', "", arr[0], flags=re.IGNORECASE) 
        if arr[0]+"_"+arr[1]+"_"+arr[3]+"_"+arr[4] in pos.keys():
            rev_pos.setdefault(pos[arr[0]+"_"+arr[1]+"_"+arr[3]+"_"+arr[4]],[]).append(arr[0]+"_"+arr[1]+"_"+arr[3]+"_"+arr[4])

print("Study samples: "+ str(len(sample_l)))
print("Study makers: "+ str(SNP_n) + "\n")
print("Choosing the interval with the max intersected SNP size ...\n")

max_v = 0;
max_interv = "";
for i in rev_pos.keys():
    if len(rev_pos[i]) > max_v :
        max_v = len(rev_pos[i])
        max_interv = i

form_interv = max_interv.split('/')

base = {'A': 0, 'T': 0.1, 'C': 0.3, 'G':0.7}
study_geno = {}
with open(sys.argv[1]) as f:
    for line in f:
        if re.match('#', line):
            continue
        arr = line.strip('\n').split('\t', 6)[0:5]
        arr[0] = re.sub('chr', "", arr[0], flags=re.IGNORECASE)
        if arr[0] != form_interv[0]:
            continue
        if arr[0]+"_"+arr[1]+"_"+arr[3]+"_"+arr[4] in rev_pos[max_interv]:
            arr = line.strip('\n').split('\t')
            index = 0
            for i in range(9, len(arr)):
                if re.match('\.', arr[i]):
                    index = 1
                    break
            if index == 1:
                continue
            allele = [arr[3], arr[4]]
            for i in range(9, len(arr)):
                arr[i] = re.sub(':.*', "", arr[i])
                arr[i] = base[allele[int(re.split(r'[\|\/]', arr[i])[0])]] + base[allele[int(re.split(r'[\|\/]', arr[i])[1])]]
            study_geno[arr[0]+"_"+arr[1]+"_"+arr[3]+"_"+arr[4]] = arr[9:len(arr)]

print(str(form_interv[0])+":"+str(form_interv[1])+"_net is chosen. The number of intersection of SNPs between study and reference in this region is "+str(len(study_geno))+"/4000\n")

study_G = np.empty(shape=[0, len(sample_l)],dtype=float)
remove_i = {}
num = 0
with open(hgimp_path + "/1KGP_CNN_net/chr" + max_interv + ".pos") as f:
    for line in f:
        geno = line.strip('\n').split('\t')
        if geno[0]+"_"+geno[1]+"_"+geno[2]+"_"+geno[3] in study_geno.keys():
            study_G = np.append(study_G, [study_geno[geno[0]+"_"+geno[1]+"_"+geno[2]+"_"+geno[3]]], axis=0)
        else:
            remove_i[num] = geno[2]
            study_G = np.append(study_G, [[base[geno[2]]+base[geno[2]]]*len(sample_l)], axis=0)
        num = num + 1

study_G = study_G.T

ref_G = np.loadtxt(hgimp_path + "/1KGP_CNN_net/chr" + max_interv + ".matrix.bz2")
for i in range(ref_G.shape[1]):
    if i in remove_i.keys():
        ref_G[:,i] = base[remove_i[i]]+base[remove_i[i]]

print("Retraining RefRGim ...\n")
snp_num = ref_G.shape[1]
y_data = np.loadtxt(hgimp_path + "/1KGP_CNN_net/chr" + form_interv[0] + "/all.subp.panel.matrix")

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
    y_pre = sess.run(prediction, feed_dict={con_out: v_xs, keep_prob: 1, phase_train: False})
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    result = sess.run(accuracy, feed_dict={con_out: v_xs, ys: v_ys, keep_prob: 1, phase_train: False})
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
            #tf.summary.histogram(".", Weights)
        with tf.name_scope('biases'):
            biases = bias_variable([out_size])
            #tf.summary.histogram(".", biases)
        with tf.name_scope('conv2d'):
            wx_plus_b = conv2d(inputs, Weights) +  biases
        with tf.name_scope('bn'):
            wx_plus_b_bn = batch_norm(wx_plus_b, out_size, phase_train)
        with tf.name_scope('activation'):
            act = activation_function(wx_plus_b_bn)
            #tf.summary.histogram('.', act)
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
con_out = tf.placeholder(tf.float32, [None, 13, 128])

with tf.name_scope('input_reshape'):
    x_image = tf.reshape(xs, [-1, 1, snp_num, 1])
    
con_result = tf.reshape(con_out, [-1, 1, 13, 128])

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
        tf.summary.histogram('.', W_fc1)
    with tf.name_scope('biases'):
        b_fc1 = bias_variable([512])
        tf.summary.histogram('.', b_fc1)
    with tf.name_scope('flat'):
        h_pool2_flat = tf.reshape(con_out, [-1, 13*128])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    with tf.name_scope('dropout'):
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope('fc2'):
    ## fc2 layer
    with tf.name_scope('weights'):
        W_fc2 = weight_variable([512, 26])
        tf.summary.histogram('.', W_fc2)
    with tf.name_scope('biases'):
        b_fc2 = bias_variable([26])
        tf.summary.histogram('.', b_fc2)
    with tf.name_scope('prediction'):
        prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

with tf.name_scope('cross_entropy'):
    with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction + 1e-10), reduction_indices=[1]))    
tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


sess = tf.Session()
path1 = sys.argv[2] + "_training_info/"
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(path1, sess.graph)
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()

saver.restore(sess, hgimp_path + "/1KGP_CNN_net/chr" + max_interv + "_net/save_net.ckpt")


for i in range(5000000):
    batch_xs, batch_ys, test_xs, test_ys = next_batch(200, ref_G, y_data)
    r1 = sess.run(layer_spp, feed_dict={xs: batch_xs, keep_prob: 1, phase_train: False}) 
    sess.run(train_step, feed_dict={con_out: r1, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        print("--- Training Round:", i, "---")
        train_acc = compute_accuracy(r1, batch_ys)
        print("Training accuracy:", train_acc)
        r2 = sess.run(layer_spp, feed_dict={xs: test_xs, keep_prob: 1, phase_train: False}) 
        test_acc = compute_accuracy(r2, test_ys)
        print("Testing accuracy:",test_acc)
        merge_result = sess.run(merged, feed_dict={con_out: r1, ys: batch_ys, keep_prob: 0.5})
        writer.add_summary(merge_result, i)
        if test_acc > 0.99 and train_acc > 0.99:
            rr = sess.run(layer_spp, feed_dict={xs: ref_G, keep_prob: 1, phase_train: False}) 
            all_acc = compute_accuracy(rr, y_data)
            if all_acc > 0.99:
                break


path2 = sys.argv[2] + "_net/save_net.ckpt"
save_path = saver.save(sess, path2)
print("Saving new CNN net to path: ", save_path + "\n")

print("Making prediction on "+ str(len(sample_l))+" input individuals ...\n")

if study_G.ndim == 1:
    study_G = np.array([study_G])

output_1 = sess.run(layer_spp, feed_dict={xs: study_G, keep_prob: 1, phase_train: False})
output_2 = sess.run(prediction, feed_dict={con_out: output_1, keep_prob: 1, phase_train: False})
popu = np.array([["ACB", "AFR"], ["ASW", "AFR"], ["BEB", "SAS"], ["CDX", "EAS"], ["CEU", "EUR"], ["CHB", "EAS"], ["CHS", "EAS"], ["CLM", "AMR"], ["ESN", "AFR"], ["FIN", "EUR"], ["GBR", "EUR"], ["GIH", "SAS"], ["GWD", "AFR"], ["IBS", "EUR"], ["ITU", "SAS"], ["JPT", "EAS"], ["KHV", "EAS"], ["LWK", "AFR"], ["MSL", "AFR"], ["MXL", "AMR"], ["PEL", "AMR"], ["PJL", "SAS"], ["PUR", "AMR"], ["STU", "SAS"], ["TSI", "EUR"], ["YRI", "AFR"]])
sample_l = np.array(sample_l).reshape(len(sample_l),1)
output_popu = np.append(sample_l, popu[np.argmax(output_2,1)], axis=1)
np.savetxt(sys.argv[2]+".populations", output_popu, delimiter='\t', fmt='%s')    
np.savetxt(sys.argv[2]+".population.probs", output_2, delimiter='\t', fmt='%1.2f')    
print("RefRGim is done. Having a nice day.")
