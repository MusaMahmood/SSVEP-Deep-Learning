# MUSA MAHMOOD - Copyright 2017
# Python 3.6.1
# TF 1.2.1

# IMPORTS:
import tensorflow as tf
import os.path as path
import itertools as it
import pandas as pd
import numpy as np
import os as os
import datetime
import glob
import time

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

# CONSTANTS:
TIMESTAMP_START = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H.%M.%S')
VERSION_NUMBER = 'v0.1.0'
DESCRIPTION_TRAINING_DATA = '_allset_'
# TRAINING_FOLDER_PATH = r'_data/my_data/S1copy/a'
TRAINING_FOLDER_PATH = r'_data/my_data/Robbie_2ch_2018_01_05'
# TEST_FOLDER_PATH = r'_data/my_data/S1copy/b'
TEST_FOLDER_PATH = r'_data/my_data/Robbie_2ch_2018_01_05/v'
EXPORT_DIRECTORY = 'model_exports/' + VERSION_NUMBER + '/'
MODEL_NAME = 'ssvep_net_2ch'
KEY_DATA_DICTIONARY = 'relevant_data'
NUMBER_STEPS = 10000
TRAIN_BATCH_SIZE = 256
TEST_BATCH_SIZE = 64
DATA_WINDOW_SIZE = 400
MOVING_WINDOW_SHIFT = 32
NUMBER_DATA_CHANNELS = 2
LEARNING_RATE = 1e-5  # 'Step size' on n-D optimization plane

NUMBER_CLASSES = 5

# FOR MODEL DESIGN
STRIDE_CONV2D = [1, 1, 1, 1]

MAX_POOL_KSIZE = [1, 2, 1, 1]
MAX_POOL_STRIDE = [1, 2, 1, 1]

BIAS_VAR_CL1 = 64
BIAS_VAR_CL2 = 128

WEIGHT_VAR_CL1 = [NUMBER_CLASSES, NUMBER_DATA_CHANNELS, 1, BIAS_VAR_CL1]  # [5, NUMBER_DATA_CHANNELS, 1, 32]
WEIGHT_VAR_CL2 = [NUMBER_CLASSES, NUMBER_DATA_CHANNELS, BIAS_VAR_CL1, BIAS_VAR_CL2]  # [5, NUMBER_DATA_CHANNELS, 32, 64]

WEIGHT_VAR_FC1 = [(DATA_WINDOW_SIZE // 4) * NUMBER_DATA_CHANNELS * BIAS_VAR_CL2, BIAS_VAR_CL1 ** 2]
MAX_POOL_FLAT_SHAPE_FC1 = [-1, NUMBER_DATA_CHANNELS * (DATA_WINDOW_SIZE // 4) * BIAS_VAR_CL2]
BIAS_VAR_FC1 = [(BIAS_VAR_CL1 ** 2)]

WEIGHT_VAR_FC_OUTPUT = [*BIAS_VAR_FC1, NUMBER_CLASSES]
BIAS_VAR_FC_OUTPUT = [NUMBER_CLASSES]

# Start Script Here:
output_folder_name = 'exports'
if not path.exists(output_folder_name):
    os.mkdir(output_folder_name)
input_node_name = 'input'
keep_prob_node_name = 'keep_prob'
output_node_name = 'output'


# Data Loading/Saving Methods:
def moving_window(data, length, step):
    # Prepare windows of 'length'
    streams = it.tee(data, length)
    # Use step of step, but don't skip any (overlap)
    return zip(*[it.islice(stream, i_, None, step) for stream, i_ in zip(streams, it.count(step=1))])


def separate_data(input_data):
    data_window_list = list(moving_window(input_data, DATA_WINDOW_SIZE, MOVING_WINDOW_SHIFT))
    shape = np.asarray(data_window_list).shape
    x_list = []
    y_list = []
    for data_window in data_window_list:
        data_window_array = np.asarray(data_window)
        count_match = np.count_nonzero(data_window_array[:, NUMBER_DATA_CHANNELS] ==
                                       data_window_array[0, NUMBER_DATA_CHANNELS])
        if count_match == shape[1]:
            x_window = data_window_array[:, 0:NUMBER_DATA_CHANNELS:1]  # [0:2:1]
            # TODO: USE SAME FILTER AS IN ANDROID (C++ filt params)
            # Will need to pass through that filter in Android before feeding to model.
            mm_scale = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(x_window)
            x_window = mm_scale.transform(x_window)

            x_list.append(x_window)
            y_list.append(data_window_array[0, NUMBER_DATA_CHANNELS])

    # get unique class values and convert to dummy values
    # convert lists to arrays; convert to 32-bit floating point
    y_array = np.asarray(y_list)
    x_array = np.asarray(x_list).astype(np.float32)
    return x_array, y_array


def load_data(data_directory):
    x_train_data = np.empty([0, DATA_WINDOW_SIZE, NUMBER_DATA_CHANNELS], np.float32)
    y_train_data = np.empty([0], np.float32)
    training_files = glob.glob(data_directory + "/*.csv")
    for f in training_files:
        data_from_file = pd.read_csv(f, header=None)  # read from file
        relevant_data = pd.DataFrame.as_matrix(data_from_file)  # convert to numpy array
        x_array, y_array = separate_data(relevant_data)
        x_train_data = np.concatenate((x_train_data, x_array), axis=0)
        y_train_data = np.concatenate((y_train_data, y_array), axis=0)
    y_train_data = np.asarray(pd.get_dummies(y_train_data).values).astype(np.float32)
    # return data_array
    print("Loaded Data Shape: X:", x_train_data.shape, " Y: ", y_train_data.shape)
    return x_train_data, y_train_data


def export_model(input_node_names, output_node_name_internal):
    freeze_graph.freeze_graph(EXPORT_DIRECTORY + MODEL_NAME + '.pbtxt', None, False,
                              EXPORT_DIRECTORY + MODEL_NAME + '.ckpt', output_node_name_internal, "save/restore_all",
                              "save/Const:0", EXPORT_DIRECTORY + '/frozen_' + MODEL_NAME + '.pb', True, "")
    input_graph_def = tf.GraphDef()
    with tf.gfile.Open(EXPORT_DIRECTORY + '/frozen_' + MODEL_NAME + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())
    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def, input_node_names, [output_node_name_internal], tf.float32.as_datatype_enum)
    with tf.gfile.FastGFile(EXPORT_DIRECTORY + '/opt_' + MODEL_NAME + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("Graph Saved - Output Directories: ")
    print("1 - Standard Frozen Model:", EXPORT_DIRECTORY + '/frozen_' + MODEL_NAME + '.pb')
    print("2 - Android Optimized Model:", EXPORT_DIRECTORY + '/opt_' + MODEL_NAME + '.pb')


# Model Building Macros: #
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# Convolution and max-pooling functions
def conv2d(x_, weights):
    return tf.nn.conv2d(x_, weights, strides=STRIDE_CONV2D, padding='SAME')


def max_pool_2x2(x_):
    return tf.nn.max_pool(x_, ksize=MAX_POOL_KSIZE,
                          strides=MAX_POOL_STRIDE, padding='SAME')


# MODEL INPUT #
x = tf.placeholder(tf.float32, shape=[None, DATA_WINDOW_SIZE, NUMBER_DATA_CHANNELS], name=input_node_name)
keep_prob = tf.placeholder(tf.float32, name=keep_prob_node_name)
y = tf.placeholder(tf.float32, shape=[None, NUMBER_CLASSES])

x_input = tf.reshape(x, [-1, DATA_WINDOW_SIZE, NUMBER_DATA_CHANNELS, 1])

# first convolution and pooling
W_conv1 = weight_variable(WEIGHT_VAR_CL1)
b_conv1 = bias_variable([BIAS_VAR_CL1])

h_conv1 = tf.nn.relu(conv2d(x_input, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second convolution and pooling
W_conv2 = weight_variable(WEIGHT_VAR_CL2)
b_conv2 = bias_variable([BIAS_VAR_CL2])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# fully connected layer1,the shape of the patch should be defined
W_fc1 = weight_variable(WEIGHT_VAR_FC1)
b_fc1 = bias_variable(BIAS_VAR_FC1)

# the input should be shaped/flattened
h_pool2_flat = tf.reshape(h_pool2, MAX_POOL_FLAT_SHAPE_FC1)
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

h_fc2_drop = tf.nn.dropout(h_fc1, keep_prob)

# weight and bias of the output layer
W_fco = weight_variable(WEIGHT_VAR_FC_OUTPUT)
b_fco = bias_variable(BIAS_VAR_FC_OUTPUT)

y_conv = tf.matmul(h_fc2_drop, W_fco) + b_fco
outputs = tf.nn.softmax(y_conv, name=output_node_name)

# training and reducing the cost/loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(outputs, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

merged_summary_op = tf.summary.merge_all()

saver = tf.train.Saver()  # Initialize tf Saver

# Load Data:
x_data, y_data = load_data(TRAINING_FOLDER_PATH)
x_val_data, y_val_data = load_data(TEST_FOLDER_PATH)
# Split training set:
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.75, random_state=1)
print("samples: train batch: ", x_train.shape)
print("samples: test batch: ", x_test.shape)
print("samples: validation batch: ", x_val_data.shape)

# TRAIN ROUTINE #
init_op = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

val_step = 0
with tf.Session(config=config) as sess:
    sess.run(init_op)
    # save model as pbtxt:
    tf.train.write_graph(sess.graph_def, EXPORT_DIRECTORY, MODEL_NAME + '.pbtxt', True)

    x_0 = np.zeros([1, DATA_WINDOW_SIZE, NUMBER_DATA_CHANNELS], dtype=np.float32)
    print("Model Dimensions: ")
    print("h_conv1: ", sess.run(h_conv1, feed_dict={x: x_0, keep_prob: 1.0}).shape)
    print("h_pool1: ", sess.run(h_pool1, feed_dict={x: x_0, keep_prob: 1.0}).shape)
    print("h_conv2: ", sess.run(h_conv2, feed_dict={x: x_0, keep_prob: 1.0}).shape)
    print("h_pool2: ", sess.run(h_pool2, feed_dict={x: x_0, keep_prob: 1.0}).shape)
    print("h_pool2_flat: ", sess.run(h_pool2_flat, feed_dict={x: x_0, keep_prob: 1.0}).shape)
    print("h_fc1: ", sess.run(h_fc1, feed_dict={x: x_0, keep_prob: 1.0}).shape)
    print("h_fc1_drop: ", sess.run(h_fc2_drop, feed_dict={x: x_0, keep_prob: 1.0}).shape)
    print("y_conv: ", sess.run(y_conv, feed_dict={x: x_0, keep_prob: 1.0}).shape)

    for i in range(NUMBER_STEPS):
        offset = (i * TRAIN_BATCH_SIZE) % (x_train.shape[0] - TRAIN_BATCH_SIZE)
        batch_x_train = x_train[offset:(offset + TRAIN_BATCH_SIZE)]
        batch_y_train = y_train[offset:(offset + TRAIN_BATCH_SIZE)]
        if i % 10 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch_x_train, y: batch_y_train, keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))

        if i % 20 == 0:
            # Calculate batch loss and accuracy
            offset = (val_step * TEST_BATCH_SIZE) % (x_test.shape[0] - TEST_BATCH_SIZE)
            batch_x_val = x_test[offset:(offset + TEST_BATCH_SIZE), :, :]
            batch_y_val = y_test[offset:(offset + TEST_BATCH_SIZE), :]
            val_accuracy = accuracy.eval(feed_dict={x: batch_x_val, y: batch_y_val, keep_prob: 1.0})
            print("Validation step %d, validation accuracy %g" % (val_step, val_accuracy))
            val_step += 1

        train_step.run(feed_dict={x: batch_x_train, y: batch_y_train, keep_prob: 0.50})

    # Run test data (entire set) to see accuracy.
    test_accuracy = sess.run(accuracy, feed_dict={x: x_test, y: y_test, keep_prob: 1.0})  # original
    print("\n Testing Accuracy:", test_accuracy, "\n\n")
    # Holdout Validation Accuracy:
    print("Holdout Validation:", sess.run(accuracy, feed_dict={x: x_val_data, y: y_val_data, keep_prob: 1.0}))

    # Save First Conv Hidden layer to x CSV files: shape[3] is the # of weights
    x_0 = np.reshape(x_val_data[1, :, :], [1, DATA_WINDOW_SIZE, NUMBER_DATA_CHANNELS])
    y_0 = np.reshape(y_val_data[1, :], [1, NUMBER_CLASSES])

    h_conv1_np = sess.run(h_conv1, feed_dict={x: x_0, keep_prob: 1.0})
    image_output_folder_name = EXPORT_DIRECTORY + DESCRIPTION_TRAINING_DATA + TIMESTAMP_START + '/' + 'h_conv1/'
    os.makedirs(image_output_folder_name)
    for i in range(h_conv1_np.shape[3]):
        filename = image_output_folder_name + 'h_conv1_' + str(i) + '.csv'
        w_reshape = np.reshape(h_conv1_np[:, :, :, i], [DATA_WINDOW_SIZE, NUMBER_DATA_CHANNELS])
        pd.DataFrame(w_reshape).to_csv(filename, index=False, header=False)

    h_conv2_np = sess.run(h_conv2, feed_dict={x: x_0, keep_prob: 1.0})
    image_output_folder_name = EXPORT_DIRECTORY + DESCRIPTION_TRAINING_DATA + TIMESTAMP_START + '/' + 'h_conv2/'
    os.makedirs(image_output_folder_name)
    for i in range(h_conv2_np.shape[3]):
        filename = image_output_folder_name + 'h_conv2_' + str(i) + '.csv'
        w_reshape = np.reshape(h_conv2_np[:, :, :, i], [DATA_WINDOW_SIZE // 2, NUMBER_DATA_CHANNELS])
        pd.DataFrame(w_reshape).to_csv(filename, index=False, header=False)

    user_input = input('Export Current Model?')
    if user_input == "1" or user_input.lower() == "y":
        saver.save(sess, EXPORT_DIRECTORY + MODEL_NAME + '.ckpt')
        export_model([input_node_name, keep_prob_node_name], output_node_name)
