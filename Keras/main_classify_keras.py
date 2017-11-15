# MUSA MAHMOOD - Copyright 2017
# Python 3.6.1
# Keras v 2.0.9

# IMPORTS:
import numpy as np
import glob
import os
import keras as k
import itertools as it

from keras import Sequential
from scipy.io import loadmat, whosmat

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
MOVING_WINDOW_SHIFT = 60
# TODO: Double check this value
NUMBER_DATA_CHANNELS = 256
NUMBER_CLASSES = 5
SAMPLING_RATE = 250
WINDOW_SECONDS = 5
WINDOW_SIZE = SAMPLING_RATE * WINDOW_SECONDS


# METHODS:
def moving_window(data, length, step):
    # Prepare windows of 'length'
    streams = it.tee(data, length)
    # Use step of step, but don't skip any (overlap)
    return zip(*[it.islice(stream, i, None, step) for stream, i in zip(streams, it.count(step=1))])


def load_training_data():
    data_array = np.empty([0, NUMBER_DATA_CHANNELS + 1], np.float32)
    # str_file_path = os.path.abspath(os.curdir)
    # print(str_file_path)
    os.chdir("..")
    str_file_path = os.path.abspath(os.curdir) + DATA_FOLDER_PATH + "/*.mat"
    print(str_file_path)
    training_files = glob.glob(str_file_path)
    print("training_files: ", np.asarray(training_files))
    for f in training_files:
        list_data = whosmat(f)
        print("Filename: ", f, "\n Data: ", list_data)
        data_from_file = loadmat(f)  # Saved as mat_dict: Dictionary with variable names as keys:
        # Extract 'relevant_data':
        relevant_data = data_from_file.get(KEY_DATA_DICTIONARY)
        if data_array.shape[1] == relevant_data.shape[1]:
            data_array = np.concatenate((data_array, relevant_data), axis=0)
        print(data_array.shape)

    return data_array


def create_keras_model():
    model = Sequential()
    model.add(k.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu',
                              input_shape=(1250, 256, 1)))

    model.add(k.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(k.layers.Conv2D(64, (5, 5), activation='relu'))

    model.add(k.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(k.layers.Flatten)

    model.add(k.layers.Dense(1000, activation='relu'))

    model.add(k.layers.Dense(NUMBER_CLASSES, activation='softmax'))

    return model


def main():
    xy_data = load_training_data()
    # Build CNN in Keras:
    keras_model = create_keras_model()

    print("Terminating...")


if __name__ == '__main__':
    main()
