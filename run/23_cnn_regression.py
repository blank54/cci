'''
1D CNN: https://wikidocs.net/80437
Dense Units: https://koreapy.tistory.com/917
CNN Filters: https://gmnam.tistory.com/274
MaxPooling: https://kevinthegrey.tistory.com/142
'''

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import sys
rootpath = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1])
sys.path.append(rootpath)

from news import NewsIO, NewsFunc, NewsPath
newsio = NewsIO()
newsfunc = NewsFunc()
newspath = NewsPath()

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


def data_split(df_x, df_y, variable_list):
    x = df_x[variable_list]
    y = df_y['cci']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False, random_state=RANDOM_STATE)
    return x_train, x_test, y_train, y_test

def model_identification(CONV_FILTER, CONV_KERNEL, ACTIVATION, POOL_SIZE, DENSE_UNIT, DROPOUT, INPUT_SHAPE, LEARNING_RATE, LOSS):
    model = keras.Sequential()
    model.add(Conv1D(filters=CONV_FILTER, kernel_size=CONV_KERNEL, activation=ACTIVATION, input_shape=INPUT_SHAPE))
    model.add(MaxPooling1D(pool_size=POOL_SIZE))
    model.add(Flatten())
    model.add(Dense(DENSE_UNIT, activation=ACTIVATION))
    model.add(Dropout(DROPOUT))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss=LOSS)

    return model

def train(x_train, y_train, NUM_EPOCHS):
    history = model.fit(x_train, y_train, epochs=NUM_EPOCHS)

    return model, history


if __name__ == '__main__':
    ## Parameters
    TOPN = 1000
    RANDOM_STATE = 42

    tf.random.set_seed(RANDOM_STATE)
    LEARNING_RATE = 1e4
    NUM_EPOCHS = 20000
    ACTIVATION = 'relu'
    LOSS = 'mse'

    DO_TRAIN = True

    ## Filenames
    fname_data_norm = f'data_w-{TOPN}_norm.pk'
    fname_correlated_variables = 'correlated_variables.json'

    ## Data import
    print('============================================================')
    print('Data import')

    df_norm = newsio.load(fname_object=fname_data_norm, _type='data')
    variable_list = newsio.load_json(fname_object=fname_correlated_variables, _type='data')
    cci = newsio.load_cci(start='200502', end='201912')
    print('  | Shape of dataset: {}'.format(df_norm.shape))

    print('============================================================')
    print('Data praparation')

    x_train, x_test, y_train, y_test = data_split(df_norm, cci, variable_list)

    ## Model training
    INPUT_SHAPE = (x_train.shape[1], 1)

    results = []
    if DO_TRAIN:
        for LEARNING_RATE in [1e-3, 3e-4, 1e-4, 3e-5]:
            for CONV_FILTER in [4, 8, 16, 32, 64]:
                for CONV_KERNEL in [1, 2, 4, 8, 16]:
                    for POOL_SIZE in [1, 2, 4]:
                        for DENSE_UNIT in [8, 16, 32, 64]:
                            for DROPOUT in [0.1, 0.2, 0.3, 0.4]:
                                fdir_model = os.path.sep.join((newspath.fdir_model, f'cnn_regression/C-{CONV_FILTER}_K-{CONV_KERNEL}_P-{POOL_SIZE}_D-{DENSE_UNIT}_O-{DROPOUT}/'))
                                os.makedirs(fdir_model, exist_ok=True)

                                fname_model = f'model_TOPN-{TOPN}_L-{LEARNING_RATE}_E-{NUM_EPOCHS}.pk'
                                fname_history = f'history_TOPN-{TOPN}_L-{LEARNING_RATE}_E-{NUM_EPOCHS}.pk'

                                x_train, x_test, y_train, y_test = data_split(df_norm, cci, variable_list)
                                INPUT_SHAPE = (x_train.shape[1], 1)

                                model = model_identification(CONV_FILTER, CONV_KERNEL, ACTIVATION, POOL_SIZE, DENSE_UNIT, DROPOUT, INPUT_SHAPE, LEARNING_RATE, LOSS)
                                model, history = train(x_train, y_train, NUM_EPOCHS)

                                newsio.save(_object=model, fname_object=fname_model, _type='model', fdir_object=fdir_model)
                                newsio.save(_object=history, fname_object=fname_history, _type='model', fdir_object=fdir_model)

                                try:
                                    RMSE = mean_squared_error(y_test, model.predict(x_test))**0.5
                                except ValueError:
                                    RMSE = 9999

                                results.append((fname_model, RMSE))
                                print(f'  | RMSE: {RMSE:.03f}')
    else:
        model = newsio.load(fname_object=fname_model, _type='model')
        history = newsio.load(fname_object=fname_history, _type='model')

    print('============================================================')
    print('Evaluation')

    for fname_model, RMSE in sorted(results, key=lambda x:x[1], reverse=False):
        print(f'{fname_model} -> {RMSE:.03f}')