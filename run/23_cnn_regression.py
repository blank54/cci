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
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import sys
rootpath = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1])
sys.path.append(rootpath)

from news import NewsIO, NewsPath
newsio = NewsIO()
newspath = NewsPath()

import numpy as np
import itertools

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam


def data_split(df_x, df_y, variable_list):
    x = df_x[variable_list]
    y = df_y['cci']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False, random_state=RANDOM_STATE)
    return x_train, x_test, y_train, y_test

def model_identification(CONV_FILTER, CONV_KERNEL, POOL_SIZE, DENSE_UNIT, DROPOUT, INPUT_SHAPE, LEARNING_RATE):
    model = keras.Sequential()
    model.add(Conv1D(filters=CONV_FILTER, kernel_size=CONV_KERNEL, activation='relu', input_shape=INPUT_SHAPE))
    model.add(MaxPooling1D(pool_size=POOL_SIZE))
    model.add(Flatten())
    model.add(Dense(DENSE_UNIT, activation='relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse')

    return model


if __name__ == '__main__':
    ## Parameters
    TOPN = 1000

    RANDOM_STATE = 42
    tf.random.set_seed(RANDOM_STATE)

    CONV_FILTER_LIST = [4, 8, 16, 32, 64]
    CONV_KERNEL_LIST = [1, 2, 4, 8, 16]
    POOL_SIZE_LIST = [1, 2, 4]
    DENSE_UNIT_LIST = [8, 16, 32, 64]
    DROPOUT_LIST = [0.1, 0.2, 0.3, 0.4]
    
    NUM_EPOCHS = 20000
    LEARNING_RATE_LIST = [1e-3, 3e-4, 1e-4, 3e-5]

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
    INPUT_SHAPE = (x_train.shape[1], 1)

    ## Model training
    print('============================================================')
    print('Model development')
    results = []
    for CONV_FILTER, CONV_KERNEL, POOL_SIZE, DENSE_UNIT, DROPOUT, LEARNING_RATE in itertools.product(CONV_FILTER_LIST, CONV_KERNEL_LIST, POOL_SIZE_LIST, DENSE_UNIT_LIST, DROPOUT_LIST, LEARNING_RATE_LIST):
        fdir_model = os.path.sep.join((newspath.fdir_model, f'TOPN-{TOPN}/cnn_regression/C-{CONV_FILTER}_K-{CONV_KERNEL}_P-{POOL_SIZE}_D-{DENSE_UNIT}_O-{DROPOUT}/'))
        fname_model = f'model_L-{LEARNING_RATE}_E-{NUM_EPOCHS}.pk'
        fname_history = f'history_L-{LEARNING_RATE}_E-{NUM_EPOCHS}.pk'
        fpath_model = os.path.sep.join((fdir_model, fname_model))
        fpath_history = os.path.sep.join((fdir_model, fname_history))

        ## Model development
        if os.path.isfile(fpath_model):
            model = newsio.load(fdir_object=fdir_model, fname_object=fname_model)
            history = newsio.load(fdir_object=fdir_model, fname_object=fname_history)
        else:
            os.makedirs(fdir_model, exist_ok=True)
        
            model = model_identification(CONV_FILTER, CONV_KERNEL, POOL_SIZE, DENSE_UNIT, DROPOUT, INPUT_SHAPE, LEARNING_RATE)
            history = model.fit(x_train, y_train, epochs=NUM_EPOCHS)

            newsio.save(_object=model, fdir_object=fdir_model, fname_object=fname_model,)
            newsio.save(_object=history, fdir_object=fdir_model, fname_object=fname_history,)

        ## Evaluation
        try:
            MSE = mean_squared_error(y_test, model.predict(x_test))
        except ValueError:
            MSE = 9999

        try:
            MAPE = mean_absolute_percentage_error(y_test, model.predict(x_test))
        except ValueError:
            MAPE = 0

        results.append((fpath_model, MSE, MAPE))

        print('--------------------------------------------------')
        print('Model information')
        print(f'  | CONV_FILTER  : {CONV_FILTER}')
        print(f'  | CONV_KERNEL  : {CONV_KERNEL}')
        print(f'  | POOL_SIZE    : {POOL_SIZE}')
        print(f'  | DENSE_UNIT   : {DENSE_UNIT}')
        print(f'  | DROPOUT      : {DROPOUT}')
        print(f'  | LEARNING_RATE: {LEARNING_RATE:.,}')
        print('--------------------------------------------------')
        print('Model performance')
        print(f'  | MSE : {MSE:.03f}')
        print(f'  | MAPE: {MAPE:.03f}')

    ## Evaluation
    print('============================================================')
    print('Evaluation')

    for fpath_model, MSE, MAPE in sorted(results, key=lambda x:x[2], reverse=False):
        print(f'  | {fpath_model} -> MSE: {MSE:.03f} & MAPE: {MAPE:.03f}')