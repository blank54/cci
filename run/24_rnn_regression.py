#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

import sys
rootpath = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1])
sys.path.append(rootpath)

from news import NewsIO, NewsPath
newsio = NewsIO()
newspath = NewsPath()

import itertools

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.sequence import TimeseriesGenerator


def data_split(df_x, df_y, variable_list):
    x = df_x[variable_list]
    y = df_y['cci']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False, random_state=RANDOM_STATE)
    return x_train, x_test, y_train, y_test

def build_dataset(x_train, x_test, y_train, y_test, BATCH_SIZE):
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(BATCH_SIZE).repeat()
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)

    return train_dataset, test_dataset

def model_identification(RNN_UNIT, DROPOUT, DENSE_UNIT, LEARNING_RATE, INPUT_SHAPE):
    model = keras.Sequential()
    model.add(SimpleRNN(RNN_UNIT, activation='relu', input_shape=INPUT_SHAPE, return_sequences=True))
    model.add(Dropout(DROPOUT))
    model.add(Dense(DENSE_UNIT))
    model.add(Dropout(DROPOUT))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse')

    return model

if __name__ == '__main__':
    ## Parameters
    TOPN = 1000

    RANDOM_STATE = 42
    tf.random.set_seed(RANDOM_STATE)

    TIMESTEPS_LIST = [1, 2, 3, 4, 5, 10, 15]
    BATCH_SIZE_LIST = [4, 8, 16, 32, 64]
    RNN_UNIT_LIST = [4, 8, 16, 32, 64]
    DENSE_UNIT_LIST = [8, 16, 32, 64]
    DROPOUT_LIST = [0.1, 0.2, 0.3, 0.4]

    NUM_EPOCHS = 20000
    LEARNING_RATE_LIST = [1e-3, 3e-4, 1e-4, 3e-5]

    ## Filenames
    fname_data_norm = f'data_w-{TOPN}_norm.pk'
    fname_correlated_variables = 'correlated_variables.json'

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
    results = []

    for TIMESTEPS, BATCH_SIZE, RNN_UNIT, DENSE_UNIT, DROPOUT, LEARNING_RATE in itertools.product(TIMESTEPS_LIST, BATCH_SIZE_LIST, RNN_UNIT_LIST, DENSE_UNIT_LIST, DROPOUT_LIST, LEARNING_RATE_LIST):
        fdir_model = os.path.sep.join((newspath.fdir_model, f'TOPN-{TOPN}/rnn_regression/T-{TIMESTEPS}_B-{BATCH_SIZE}_R-{RNN_UNIT}_D-{DENSE_UNIT}_O-{DROPOUT}/'))
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

            train_generator = TimeseriesGenerator(x_train, y_train, length=TIMESTEPS, batch_size=BATCH_SIZE)
            INPUT_SHAPE = (train_generator[0][0].shape[1], train_generator[0][0].shape[2])
            model = model_identification(RNN_UNIT, DROPOUT, DENSE_UNIT, LEARNING_RATE, INPUT_SHAPE)
            history = model.fit_generator(train_generator, epochs=NUM_EPOCHS)

            newsio.save(_object=model, fname_object=fname_model, _type='model', fdir_object=fdir_model)
            newsio.save(_object=history, fname_object=fname_history, _type='model', fdir_object=fdir_model)

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
        print(f'  | TIMESTEPS    : {TIMESTEPS}')
        print(f'  | BATCH_SIZE   : {BATCH_SIZE}')
        print(f'  | RNN_UNIT     : {RNN_UNIT}')
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