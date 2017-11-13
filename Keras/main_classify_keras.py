# MUSA MAHMOOD - Copyright 2017
# Python 3.6.1
# Keras v 2.0.9

# IMPORTS:
import numpy as np
import glob
import os
import scipy as sp

from scipy.io import loadmat, whosmat

# CONSTANTS:
VERSION_NUMBER = 'v0.0.1'
DATA_FOLDER_PATH = r'/DATA/MAMEM_Small_14ch'
EXPORT_DIRECTORY = 'model_exports/' + VERSION_NUMBER + '/'
MODEL_NAME = 'ssvep_net_14ch'
NUMBER_STEPS = 5000
TRAIN_BATCH_SIZE = 256
VAL_BATCH_SIZE = 10
DATA_WINDOW_SIZE = 300
MOVING_WINDOW_SHIFT = 60
# TODO: Double check this value
NUMBER_DATA_CHANNELS = 14


# METHODS:
def load_data():
    data_array = np.empty([0, 3], np.float32)
    # str_file_path = os.path.abspath(os.curdir)
    # print(str_file_path)
    os.chdir("..")
    str_file_path = os.path.abspath(os.curdir)+DATA_FOLDER_PATH+"/*.mat"
    print(str_file_path)
    training_files = glob.glob(str_file_path)
    print("training_files: ", np.asarray(training_files))
    for f in training_files:
        list_data = whosmat(f)
        print(list_data)
        # data_from_file = loadmat(f)


def main():
    load_data()
    print("Terminating...")


if __name__ == '__main__':
    main()


