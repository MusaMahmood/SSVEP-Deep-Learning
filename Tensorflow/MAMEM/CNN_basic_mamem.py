# MUSA MAHMOOD - Copyright 2017
# Python 3.6.1
# TF 1.2.1
# NOTE: In this context 'multichannel' means more than 2.

# IMPORTS:
import numpy as np
import glob
import os
import os.path as path
import itertools as it
import tensorflow as tf
import pandas as pd
# import matplotlib.pyplot as plt

from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

# CONSTANTS:
# SPECIFIC_CHANNEL_SELECTION = np.asarray([120, 121, 122, 123, 124, 125, 126, 138, 149, 158, 167, 175, 187])
SPECIFIC_CHANNEL_SELECTION = np.asarray(range(120, 136))
NUMBER_CHANNELS_SELECT = SPECIFIC_CHANNEL_SELECTION.shape[0]  # Selects first int in shape

VERSION_NUMBER = 'v0.0.1'
DATA_FOLDER_PATH = r'/DATA/output_csv/S001_f'
KEY_DATA_DICTIONARY = 'relevant_data'
EXPORT_DIRECTORY = 'model_exports/' + VERSION_NUMBER + '/'
MODEL_NAME = 'ssvep_net_14ch'
NUMBER_STEPS = 5000
TRAIN_BATCH_SIZE = 256
VAL_BATCH_SIZE = 64
DATA_WINDOW_SIZE = 500
MOVING_WINDOW_SHIFT = 25

NUMBER_CHANNELS_TOTAL = 256
NUMBER_CLASSES = 5

# FOR MODEL DESIGN
STRIDE_CONV2D = [1, 1, 1, 1]

BIAS_VAR_CL1 = 32
BIAS_VAR_CL2 = 64

WEIGHT_VAR_CL1 = [NUMBER_CLASSES, 1, 1, BIAS_VAR_CL1]
WEIGHT_VAR_CL2 = [NUMBER_CLASSES, 1, BIAS_VAR_CL1, BIAS_VAR_CL2]

MAX_POOL_KSIZE = [1, 2, 1, 1]
MAX_POOL_STRIDE = [1, 2, 1, 1]

FC_LAYER_DIMENSIONS = (DATA_WINDOW_SIZE // 4) * BIAS_VAR_CL2
WEIGHT_VAR_FC1 = [NUMBER_CHANNELS_SELECT * FC_LAYER_DIMENSIONS, (BIAS_VAR_CL1 ** 2)]
MAX_POOL_FLAT_SHAPE_FC1 = [-1, NUMBER_CHANNELS_SELECT * FC_LAYER_DIMENSIONS]
BIAS_VAR_FC1 = [(BIAS_VAR_CL1 ** 2)]
BIAS_VAR_FC2 = [(BIAS_VAR_CL1 ** 2) * 2]
WEIGHT_VAR_FC2 = [*BIAS_VAR_FC1, *BIAS_VAR_FC2]
WEIGHT_VAR_FC_OUTPUT = [*BIAS_VAR_FC2, NUMBER_CLASSES]
BIAS_VAR_FC_OUTPUT = [NUMBER_CLASSES]


def get_data_directory():
    os.chdir("..")
    return os.path.abspath(os.curdir) + DATA_FOLDER_PATH


def moving_window(data, length, step):
    # Prepare windows of 'length'
    streams = it.tee(data, length)
    # Use step of step, but don't skip any (overlap)
    return zip(*[it.islice(stream, i, None, step) for stream, i in zip(streams, it.count(step=1))])


def separate_data(input_data):
    data_window_list = list(moving_window(input_data, DATA_WINDOW_SIZE, MOVING_WINDOW_SHIFT))
    shape = np.asarray(data_window_list).shape
    x_list = []
    y_list = []
    for data_window in data_window_list:
        data_window_array = np.asarray(data_window)
        count_match = np.count_nonzero(data_window_array[:, NUMBER_CHANNELS_TOTAL] ==
                                       data_window_array[0, NUMBER_CHANNELS_TOTAL])
        if count_match == shape[1]:
            x_window = data_window_array[:, SPECIFIC_CHANNEL_SELECTION]  # [0:2:1]
            # USE SAME FILTER AS IN ANDROID (C++ filt params),
            # Will need to pass through that filter in Android before feeding to model.
            mm_scale = preprocessing.MinMaxScaler().fit(x_window)
            x_window = mm_scale.transform(x_window)

            x_list.append(x_window)
            y_list.append(data_window_array[0, NUMBER_CHANNELS_TOTAL])

    # get unique class values and convert to dummy values
    # convert lists to arrays; convert to 32-bit floating point
    # TODO: DO THIS ELSEWHERE:
    y_array = np.asarray(y_list)
    # print("y_array.shape", y_array.shape)
    x_array = np.asarray(x_list).astype(np.float32)
    # print("x_array.shape", x_array.shape)
    return x_array, y_array


def load_data(data_directory, letters, selection):
    x_train_data = np.empty([0, DATA_WINDOW_SIZE, NUMBER_CHANNELS_SELECT], np.float32)
    y_train_data = np.empty([0], np.float32)
    for s in letters:
        str_file_path = data_directory + "/*" + s + "*.mat"
        print(str_file_path)
        training_files = glob.glob(str_file_path)
        # TODO: KEEP ONLY 'SELECTION' TRIALS:
        training_files = np.asarray(training_files)[[selection]]
        print("training_files: ", training_files)
        for f in training_files:
            data_from_file = loadmat(f)  # Saved as mat_dict: Dictionary with variable names as keys:
            # Extract 'relevant_data':
            relevant_data = data_from_file.get(KEY_DATA_DICTIONARY)
            x, y = separate_data(relevant_data)
            x_train_data = np.concatenate((x_train_data, x), axis=0)
            y_train_data = np.concatenate((y_train_data, y), axis=0)
    # return data_array
    y_train_data = np.asarray(pd.get_dummies(y_train_data).values).astype(np.float32)
    return x_train_data, y_train_data


def model_input(input_node_name, keep_prob_node_name):
    x = tf.placeholder(tf.float32, shape=[None, DATA_WINDOW_SIZE, NUMBER_CHANNELS_SELECT], name=input_node_name)
    keep_prob = tf.placeholder(tf.float32, name=keep_prob_node_name)
    y_ = tf.placeholder(tf.float32, shape=[None, NUMBER_CLASSES])
    return x, keep_prob, y_


# weights and bias functions for convolution
def weight_variable(shape):
    # Outputs random values from a truncated normal distribution:
    # Args: shape.
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# Convolution and max-pooling functions
# Computes 2-D convolution given 4-D input and filter tensors
def conv2d(x, weights):
    return tf.nn.conv2d(x, weights, strides=STRIDE_CONV2D, padding='SAME')


def max_pool_2x2(x, ksize, stride):  # Args:
    # ksize(4-D):  The size of the window for each dimension of the input tensor.
    # strides(4-D): The stride of the sliding window for each dimension of the input tensor.
    return tf.nn.max_pool(x, ksize=ksize,
                          strides=stride, padding='SAME')


def build_model(x, keep_prob, y, output_node_name):
    x_input = tf.reshape(x, [-1, DATA_WINDOW_SIZE, NUMBER_CHANNELS_SELECT, 1])  # [-1, 250, 256, 1]

    # first convolution and pooling
    w_conv1 = weight_variable(WEIGHT_VAR_CL1)
    b_conv1 = bias_variable([BIAS_VAR_CL1])
    # tf.nn.relu: Computes rectified linear: max(features, 0)
    # Args: features (a tensor)
    print("x_input.shape", x_input.shape)
    print("W_conv1.shape", w_conv1.shape)
    h_conv1 = tf.nn.relu(conv2d(x_input, w_conv1) + b_conv1)
    print("1st conv layer dimensions: ", h_conv1.shape)
    # Performs max pooling on the inputs.
    h_pool1 = max_pool_2x2(h_conv1, MAX_POOL_KSIZE, MAX_POOL_STRIDE)
    print("h_pool1 output", h_pool1.shape)
    # second convolution and pooling
    w_conv2 = weight_variable(WEIGHT_VAR_CL2)
    b_conv2 = bias_variable([BIAS_VAR_CL2])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2, MAX_POOL_KSIZE, MAX_POOL_STRIDE)

    # fully connected layer1,the shape of the patch should be defined
    w_fc1 = weight_variable(WEIGHT_VAR_FC1)
    b_fc1 = bias_variable(BIAS_VAR_FC1)

    # the input should be shaped/flattened
    h_pool2_flat = tf.reshape(h_pool2, MAX_POOL_FLAT_SHAPE_FC1)
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

    # fully connected layer2
    w_fc2 = weight_variable(WEIGHT_VAR_FC2)
    b_fc2 = bias_variable(BIAS_VAR_FC2)
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2)

    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    # weight and bias of the output layer
    w_fc_output = weight_variable(WEIGHT_VAR_FC_OUTPUT)
    b_fc_output = bias_variable(BIAS_VAR_FC_OUTPUT)

    y_conv = tf.matmul(h_fc2_drop, w_fc_output) + b_fc_output
    outputs = tf.nn.softmax(y_conv, name=output_node_name)

    # training and reducing the cost/loss function
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(outputs, 1), tf.argmax(y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    merged_summary_op = tf.summary.merge_all()

    return train_step, cross_entropy, accuracy, merged_summary_op


def train_and_test(x_train_data, y_train_data, x_test_data, y_test_data, x, keep_prob, y, train_step, accuracy, saver):
    val_step = 0
    # split into data windows & store:
    x_train, x_test, y_train, y_test = train_test_split(x_train_data, y_train_data, train_size=0.75, random_state=1)
    print("train_split x:", x_train.shape, " y: ", y_train.shape)
    print("test_split x:", x_test.shape, " y: ", y_test.shape)
    init_op = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        # save model as pbtxt:
        tf.train.write_graph(sess.graph_def, EXPORT_DIRECTORY, MODEL_NAME + '.pbtxt', True)
        print("TRAIN_INPUT_SIZE: = ", TRAIN_BATCH_SIZE, "x", x_train.shape[1:3:1])
        print("VAL_INPUT_SIZE: = ", VAL_BATCH_SIZE, "x", x_train.shape[1:3:1])
        for i in range(NUMBER_STEPS):
            offset = (i * TRAIN_BATCH_SIZE) % (x_train.shape[0] - TRAIN_BATCH_SIZE)
            batch_x_train = x_train[offset:(offset + TRAIN_BATCH_SIZE)]
            # shape_original = batch_x_train.shape

            batch_y_train = y_train[offset:(offset + TRAIN_BATCH_SIZE)]
            if i % 10 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch_x_train, y: batch_y_train, keep_prob: 1.0})
                print("step %d, training accuracy %g" % (i, train_accuracy))

            if i % 20 == 0:
                # Calculate batch loss and accuracy
                offset = (val_step * VAL_BATCH_SIZE) % (x_test.shape[0] - VAL_BATCH_SIZE)
                batch_x_val = x_test[offset:(offset + VAL_BATCH_SIZE), :, :]
                # shape_original = batch_x_val.shape
                batch_y_val = y_test[offset:(offset + VAL_BATCH_SIZE), :]
                val_accuracy = accuracy.eval(feed_dict={x: batch_x_val, y: batch_y_val, keep_prob: 1.0})
                print("Validation step %d, validation accuracy %g" % (val_step, val_accuracy))
                val_step += 1

            train_step.run(feed_dict={x: batch_x_train, y: batch_y_train, keep_prob: 0.25})
        # shape_original = x_test.shape
        test_accuracy = sess.run(accuracy, feed_dict={x: x_test, y: y_test, keep_prob: 1.0})  # original
        # test_accuracy = sess.run(accuracy, feed_dict={x: x_test, y: y_test, keep_prob: 0.5})
        print("\n Validation Accuracy (full):", test_accuracy, "\n\n")

        # save temp checkpoint
        saver.save(sess, EXPORT_DIRECTORY + MODEL_NAME + '.ckpt')

        # run Test:
        print("Test Accuracy:",
              sess.run(accuracy, feed_dict={x: x_test_data, y: y_test_data, keep_prob: 1.0}))


def export_model(input_node_names, output_node_name):
    freeze_graph.freeze_graph(EXPORT_DIRECTORY + MODEL_NAME + '.pbtxt', None, False,
                              EXPORT_DIRECTORY + MODEL_NAME + '.ckpt', output_node_name, "save/restore_all",
                              "save/Const:0", EXPORT_DIRECTORY + '/frozen_' + MODEL_NAME + '.pb', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open(EXPORT_DIRECTORY + '/frozen_' + MODEL_NAME + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def, input_node_names, [output_node_name], tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile(EXPORT_DIRECTORY + '/opt_' + MODEL_NAME + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("Graph Saved - Output Directories: ")
    print("1 - Standard Frozen Model:", EXPORT_DIRECTORY + '/frozen_' + MODEL_NAME + '.pb')
    print("2 - Android Optimized Model:", EXPORT_DIRECTORY + '/opt_' + MODEL_NAME + '.pb')


def main():
    # Configure Export Folder for Model:
    output_folder_name = 'exports'
    if not path.exists(output_folder_name):
        os.mkdir(output_folder_name)

    input_node_name = 'input'
    keep_prob_node_name = 'keep_prob'
    output_node_name = 'output'
    x, keep_prob, y_ = model_input(input_node_name, keep_prob_node_name)
    train_step, loss, accuracy, merged_summary_op = build_model(x, keep_prob, y_, output_node_name)
    saver = tf.train.Saver()
    data_directory = get_data_directory()
    x_train_data, y_train_data = load_data(data_directory, ['a'], np.s_[8:24])  # 9, 10, 12, 13, 15, 16, 18, 19, 21, 22
    print("Training Data: X:", x_train_data.shape, " Y: ", y_train_data.shape)
    x_test_data, y_test_data = load_data(data_directory, ['a'], np.s_[0:9])
    print("Test Data: X:", x_test_data.shape, " Y: ", y_test_data.shape)
    train_and_test(x_train_data, y_train_data, x_test_data, y_test_data, x, keep_prob, y_, train_step, accuracy, saver)
    user_input = input('Export Current Model?')
    if user_input == "1" or user_input.lower() == "y":
        export_model([input_node_name, keep_prob_node_name], output_node_name)
    print("Terminating...")


if __name__ == '__main__':
    main()
