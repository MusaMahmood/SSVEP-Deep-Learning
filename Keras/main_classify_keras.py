# MUSA MAHMOOD - Copyright 2017
# Python 3.6.1
# Keras v 2.0.9

# IMPORTS:
import numpy as np
import glob
import os
import keras as k
import itertools as it
import tensorflow as tf

from sklearn import preprocessing
from keras import Sequential
from scipy.io import loadmat

# CONSTANTS:
VERSION_NUMBER = 'v0.0.1'
DATA_FOLDER_PATH = r'/DATA/output_csv/S001'
KEY_DATA_DICTIONARY = 'relevant_data'
EXPORT_DIRECTORY = 'model_exports/' + VERSION_NUMBER + '/'
MODEL_NAME = 'ssvep_net_14ch'
NUMBER_STEPS = 5000
TRAIN_BATCH_SIZE = 256
VAL_BATCH_SIZE = 10
DATA_WINDOW_SIZE = 300
MOVING_WINDOW_SHIFT = 150

NUMBER_CHANNELS_SELECT = 16
SPECIFIC_CHANNEL_SELECTION = range(NUMBER_CHANNELS_SELECT)
NUMBER_CHANNELS_TOTAL = 256

FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.1  # starting value of epsilon

# TODO: Double check this value
NUMBER_DATA_CHANNELS = 256
NUMBER_CLASSES = 5


# METHODS:
def get_data_directory():
    os.chdir("..")
    return os.path.abspath(os.curdir) + DATA_FOLDER_PATH


def moving_window(data, length, step):
    # Prepare windows of 'length'
    streams = it.tee(data, length)
    # Use step of step, but don't skip any (overlap)
    return zip(*[it.islice(stream, i, None, step) for stream, i in zip(streams, it.count(step=1))])


def separate_data(input_data, total_number_channels):
    data_window_list = list(moving_window(input_data, DATA_WINDOW_SIZE, MOVING_WINDOW_SHIFT))
    shape = np.asarray(data_window_list).shape
    print("dataWindowList.shape (windows, window length, columns)", shape)
    x_list = []
    y_list = []
    for data_window in data_window_list:
        data_window_array = np.asarray(data_window)
        count_match = np.count_nonzero(data_window_array[:, total_number_channels] ==
                                       data_window_array[0, total_number_channels])
        if count_match == shape[1]:
            x_window = data_window_array[:, SPECIFIC_CHANNEL_SELECTION]  # [0:2:1]
            # USE SAME FILTER AS IN ANDROID (C++ filt params),
            # Will need to pass through that filter in Android before feeding to model.
            mm_scale = preprocessing.MinMaxScaler().fit(x_window)
            x_window = mm_scale.transform(x_window)
            x_list.append(x_window)
            y_list.append(data_window_array[0, total_number_channels])

    # get unique class values and convert to dummy values
    # convert lists to arrays; convert to 32-bit floating point
    # y_array = np.asarray(pd.get_dummies(y_list).values).astype(np.float32)
    y_array = np.asarray(y_list).astype(np.float32)
    y_array = np.reshape(y_array, [*y_array.shape, -1])
    print("y_array.shape", y_array.shape)
    x_array = np.asarray(x_list).astype(np.float32)
    print("x_array.shape", x_array.shape)
    return x_array, y_array


def load_data(data_directory, letters):
    number_channels = 0
    data_array = np.empty([0, NUMBER_CHANNELS_TOTAL + 1], np.float32)
    for s in letters:
        str_file_path = data_directory + "/*" + s + "*.mat"
        print(str_file_path)
        training_files = glob.glob(str_file_path)
        for f in training_files:
            data_from_file = loadmat(f)  # Saved as mat_dict: Dictionary with variable names as keys:
            relevant_data = data_from_file.get(KEY_DATA_DICTIONARY)
            if data_array.shape[1] == relevant_data.shape[1]:
                data_array = np.concatenate((data_array, relevant_data), axis=0)
            number_channels = data_array.shape[1] - 1  # Subtract for labels column.
            # print("Number channels: ", number_channels)
        print(data_array.shape)
    return separate_data(data_array, number_channels)


def create_keras_model(input_shape, kernel_size):
    print("kernel_size: ", kernel_size)
    model = Sequential()
    model.add(k.layers.Conv2D(32, kernel_size=kernel_size, strides=(1, 1), activation='relu', input_shape=input_shape))
    model.add(k.layers.MaxPooling2D(pool_size=(1, 1), strides=(1, 1)))
    model.add(k.layers.Conv2D(64, kernel_size, activation='relu'))
    model.add(k.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(k.layers.Flatten())
    model.add(k.layers.Dense(512, activation='relu'))
    model.add(k.layers.Dense(NUMBER_CLASSES+1, activation='softmax'))
    adam = k.optimizers.Adam(lr=1e-4)
    model.compile(loss="mse", optimizer=adam, metrics=['accuracy'])
    model.summary()
    return model


def train_network(model, data, labels):
    print("Train...")
    print("ydata_shape: ", labels.shape)
    # exit()
    data = np.reshape(data, [*data.shape, 1])
    print("xdata_shape: ", data.shape)

    labels_ker = k.utils.to_categorical(labels)
    model.fit(data, labels_ker, epochs=10, batch_size=32)
    model.save_weights(EXPORT_DIRECTORY)


def main():
    data_directory = get_data_directory()
    x_data, y_data = load_data(data_directory, ['a', 'b', 'c', 'd'])
    # Build CNN in Keras:
    input_shape = [*x_data.shape[1:3:1], 1]
    print("input_shape: ", x_data.shape)
    # for i in range(5):
    #     keras_model = create_keras_model(input_shape, [i, i])
    keras_model = create_keras_model(input_shape, [2, 2])
    train_network(keras_model, x_data, y_data)
    print("Terminating...")


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.DEBUG)
    K.set_session(sess)
    main()
