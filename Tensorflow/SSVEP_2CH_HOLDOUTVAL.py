import pandas as pd
import os
import matplotlib.pyplot as plt
import glob
from sklearn import preprocessing
import numpy as np
import itertools as it
import tensorflow as tf
from sklearn.model_selection import train_test_split

# from sklearn.model_selection import KFold, cross_val_score

pathtr = r'_data/S1copy/a/'
pathvalte = r'_data/S1copy/b/'
dflisttr = []
dflistvaltest = []
labels = []
Xtrain_ = []
Ytrain_ = []
Xtestval_ = []
Ytestval_ = []
trainbatch_size = 128
valbatch_size = 128
trainingfiles = glob.glob(pathtr + "/*.csv")
valtesfiles = glob.glob(pathvalte + "/*.csv")

for filenametr in trainingfiles:
    df = pd.read_csv(filenametr, header=None)
    dflisttr.append(df)

for filenamevalte in valtesfiles:
    df = pd.read_csv(filenamevalte, header=None)
    dflistvaltest.append(df)

traindata = pd.concat(dflisttr, axis=0)
testvaldata = pd.concat(dflistvaltest, axis=0)

traindata_ = np.asarray(traindata)
testvaldata_ = np.asarray(testvaldata)


def moving_window(data, length, step):
    # Prepare windows of 'length'
    streams = it.tee(data, length)
    # Use step of step, but don't skip any (overlap)
    return zip(*[it.islice(stream, i, None, step) for stream, i in zip(streams, it.count(step=1))])


# def moving_window(data_, length, step=1):
#     streams = it.tee(data_, length)
#     return zip(*[it.islice(stream, i, None, step * length) for stream, i in zip(streams, it.count(step=step))])

traindata_ = list(moving_window(traindata_, 400, 60))
traindata_ = np.asarray(traindata_)
for i in traindata_:
    labeltrain = np.all(i == i[0, :], axis=0)

    if labeltrain[2] == True:
        xtr = i[:, 0:2]
        minmax_scale = preprocessing.MinMaxScaler().fit(xtr)
        Xtr = minmax_scale.transform(xtr)
        Xtrain_.append(Xtr)
        Ytrain_.append(i[0, 2])

ytrain = pd.get_dummies(Ytrain_)
ytrain = ytrain.values
Ytrain = np.asarray(ytrain)
Xtrain = np.asarray(Xtrain_)
Xtrain = Xtrain.astype(np.float32)
Ytrain = Ytrain.astype(np.float32)

testvaldata_ = list(moving_window(testvaldata_, 400, 60))
testvaldata_ = np.asarray(testvaldata_)

for j in testvaldata_:
    labeltestval = np.all(j == j[0, :], axis=0)

    if labeltestval[2]:
        xtestval = j[:, 0:2]
        minmax_scale = preprocessing.MinMaxScaler().fit(xtestval)
        Xtesval = minmax_scale.transform(xtestval)
        Xtestval_.append(Xtesval)
        Ytestval_.append(j[0, 2])

ytestval = pd.get_dummies(Ytestval_)
ytestval = ytestval.values
Ytestval = np.asarray(ytestval)
Xtestval = np.asarray(Xtestval_)
Xtesval = Xtestval.astype(np.float32)
Ytestval = Ytestval.astype(np.float32)

# Starting the Session
sess = tf.InteractiveSession()

# creating plaaceholders
x = tf.placeholder(tf.float32, shape=[None, 400, 2])
y = tf.placeholder(tf.float32, shape=[None, 5])


# weights and bias functions for convolution
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# convoluton and maxpooling functions
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 1, 1],
                          strides=[1, 2, 1, 1], padding='SAME')


# reshape in input
x_input = tf.reshape(x, [-1, 400, 2, 1])

# first convolution and pooling
W_conv1 = weight_variable([10, 1, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_input, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second convolution and pooling
W_conv2 = weight_variable([10, 1, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# fully connected layer,the shape of the patch should be defined
W_fc1 = weight_variable([100 * 2 * 64, 1024])
b_fc1 = bias_variable([1024])

# the input should be shaped/flattened
h_pool2_flat = tf.reshape(h_pool2, [-1, 100 * 2 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# weight and bias of the output layer
W_fc2 = weight_variable([1024, 5])
b_fc2 = bias_variable([5])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# training and reducing the cost/loss fucntion
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))

# Accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

val_step = 0

for i in range(5000):
    offset = (i * trainbatch_size) % (Xtrain.shape[0] - trainbatch_size)
    batch_xtrain = Xtrain[offset:(offset + trainbatch_size)]
    batch_ytrain = Ytrain[offset:(offset + trainbatch_size)]
    if i % 10 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch_xtrain, y: batch_ytrain, keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))

    if i % 100 == 0:
        # Calculate batch loss and accuracy
        offset = (val_step * valbatch_size) % (Xtesval.shape[0] - valbatch_size)
        batch_xval = Xtesval[offset:(offset + valbatch_size), :, :]
        batch_yval = Ytestval[offset:(offset + valbatch_size), :]
        val_accuracy = accuracy.eval(feed_dict={x: batch_xval, y: batch_yval, keep_prob: 1.0})
        print("Validation step %d, validation accuracy %g" % (val_step, val_accuracy))
        val_step += 1

    # train_step.run(feed_dict={x: batch_xtrain, y: batch_ytrain, keep_prob: 0.5})
    train_step.run(feed_dict={x: batch_xtrain, y: batch_ytrain, keep_prob: 0.5})

    # Calculate accuracy for 100 test data
print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: Xtesval[0:128], y: Ytestval[0:128], keep_prob: 1.0}))

# saver=tf.train.Saver()
# save_path='trainedmodel/georgiatek'
# saver.save(sess=sess,save_path=save_path)
# saver.restore(sess=sess,save_path=save_path)
# print('Saved trained model Accuracy:',sess.run(accuracy,feed_dict={x:Xtestval,y:Ytestval,keep_prob:1}))
