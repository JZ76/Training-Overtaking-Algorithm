"""
Copyright 2022 Jiancheng Zhang

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.python.keras.models import load_model, Model
from tensorflow.python.keras.layers import LSTM, Dense, Input, Masking, SimpleRNN, concatenate

import os
import re
import gc

"""
A lot of code is same in Colab_version.ipynb, so the comment is omitted for same code, but I commented different part
"""


# ----------------------------------------------------------------------------------------------------------------------
def loading_data(folder):
    check_name = re.compile('^ML')

    datasets = []

    # https://stackoverflow.com/questions/4813061/non-alphanumeric-list-order-from-os-listdir
    for filename in sorted(os.listdir(folder)):

        files = os.path.join(folder, filename)

        if re.match(check_name, filename) and os.path.isfile(files):
            temp = pd.read_csv(files, skiprows=1, header=None).iloc[:, :-1]

            temp[0] = temp[0].map(lambda t: t / 16.0)
            temp[1] = temp[1].map(lambda t: t / 0.192)

            datasets.append(np.array(temp, dtype=float))

    gc.collect()

    return datasets


# ----------------------------------------------------------------------------------------------------------------------
def loading_speed_steering_data(folder):
    check_name = re.compile('^car_state_blue')

    datasets = []

    for filename in sorted(os.listdir(folder)):

        files = os.path.join(folder, filename)

        if re.match(check_name, filename) and os.path.isfile(files):
            datasets.append(np.array(pd.read_csv(files).iloc[:, [3, 5]], dtype=float))

    gc.collect()

    return datasets


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
# ----------------------------------------------------------------------------------------------------------------------
# loading dataset ------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

    folder = "/path/to/your/dataset"

    datasets = loading_data(folder)
    speed_steering = loading_speed_steering_data(folder)

    new_datasets = datasets
    new_speed_steering = speed_steering

    X = []
    y = []
    # I cannot find tensorflow.keras.preprocessing.sequence.pad_sequences() function in macOS version
    # So, I am going to pad dataset manually
    # We need to know the max size in the whole dataset
    max_seq_len = 0

    for x in new_datasets:
        X.append(x[:, 2:])
        y.append(x[:, 0:2])
        # shape[0] is how many timestamps in this instance, shape[1] is 1083
        max_seq_len = max(max_seq_len, x.shape[0])

# ----------------------------------------------------------------------------------------------------------------------
# manually pad the dataset ---------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# https://datascience.stackexchange.com/questions/48796/how-to-feed-lstm-with-different-input-array-sizes

    special_value = -100.0

    Xpad = []
    ypad = []

    # creating two lists that have same number of instances as new_datasets,
    # each instance is a numpy array that full of special_value, and shape is max_seq_len times 1081 and 2
    # this makes all the instance have same size
    for i in range(len(new_datasets)):
        Xpad.append(np.full((max_seq_len, 1081), fill_value=special_value))
        ypad.append(np.full((max_seq_len, 2), fill_value=special_value))

    # we can start copying data from X and y to Xpad and ypad
    # s is the index, x is the data
    for s, x in enumerate(X):
        # get the size of this instance (original)
        seq_len = x.shape[0]
        # copy data from the original instance to new instance
        # due to first dimension is list, second and third dimension is numpy array.
        # So, we need two square brackets
        # after copying, 0 to seq_len is the original data, and seq_len to max_seq_len is special_value
        Xpad[s][0:seq_len, :] = x

    for s, x in enumerate(y):
        seq_len = x.shape[0]
        ypad[s][0:seq_len, :] = x

    # need to convert to numpy array, because the first dimension is a list
    # the result is same as using tensorflow.keras.preprocessing.sequence.pad_sequences()
    Xpad_A = np.asarray(Xpad)
    ypad = np.asarray(ypad)

    datasets = []
    new_datasets = []
    X = []
    y = []
    Xpad = []
    gc.collect()

# ----------------------------------------------------------------------------------------------------------------------
    speed_steering_Pad = []

    for i in range(len(new_speed_steering)):
        speed_steering_Pad.append(np.full((max_seq_len, 2), fill_value=special_value))

    for s, x in enumerate(new_speed_steering):
        seq_len = x.shape[0]
        speed_steering_Pad[s][0:seq_len, :] = x

    XPad_B = np.asarray(speed_steering_Pad)

    speed_steering = []
    new_speed_steering = []
    speed_steering_Pad = []
    gc.collect()

# ----------------------------------------------------------------------------------------------------------------------
# Option A build a new model -------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# https://stackoverflow.com/questions/46982616/batch-input-shape-tuple-on-keras-lstm

    inputA = Input(shape=(None, 1081))
    A = Masking(input_shape=(None, 1081), mask_value=special_value)(inputA)
    x = Dense(500, activation="relu")(A)
    x = SimpleRNN(150, return_sequences=True, input_shape=(None, 1081))(x)

    inputB = Input(shape=(None, 2))
    B = Masking(input_shape=(None, 2), mask_value=special_value)(inputB)

    combined = concatenate([x, B], axis=2)

    z = Dense(256, activation="relu")(combined)
    z = Dense(128, activation="relu")(z)
    z = Dense(32, activation="tanh")(z)
    z = Dense(2, activation="tanh")(z)

    SimpleRNN_model = Model(inputs=[inputA, inputB], outputs=z)

# ----------------------------------------------------------------------------------------------------------------------
# Option B load an old model -------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

    # SimpleRNN_model = load_model('/path/to/your/model')

# ----------------------------------------------------------------------------------------------------------------------
# Start training -------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

    SimpleRNN_model.compile(loss="mean_squared_error", optimizer="RMSprop", metrics=['mean_squared_error'])
    SimpleRNN_model.summary()

    # https://github.com/tensorflow/tensorflow/issues/56082
    # you can open Activity Monitor->Window->GPU History to see your GPU usage during training
    with tf.device("/device:CPU:0"):
        SimpleRNN_model.fit([Xpad_A, XPad_B], ypad, epochs=10, batch_size=5, verbose=1, shuffle=True)

    SimpleRNN_model.save("/path/to/your/model")
