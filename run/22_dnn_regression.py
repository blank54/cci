'''
DNN regression: https://ysyblog.tistory.com/101
Early stopping: https://3months.tistory.com/424
NaN loss: https://whiteglass.tistory.com/1
'''

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
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


def data_split(df_x, df_y, variable_list):
    x = df_x[variable_list]
    y = df_y['cci']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False, random_state=RANDOM_STATE)
    return x_train, x_test, y_train, y_test

def build_dataset(x_train, x_test, y_train, y_test, BATCH_SIZE):
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(BATCH_SIZE).repeat()
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)

    return train_dataset, test_dataset

def model_identification(INPUT_SHAPE, DROPOUT, LEARNING_RATE):
    model = keras.Sequential()
    model.add(Dense(units=16, activation='relu', input_shape=INPUT_SHAPE))
    model.add(Dropout(DROPOUT))
    model.add(Dense(units=8, activation='relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(units=4, activation='relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(units=2, activation='relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(units=1))

    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse')

    return model

def train(model, train_dataset, test_dataset, NUM_EPOCHS, STEPS_PER_EPOCH, VALIDATION_STEPS):
    history = model.fit(train_dataset,
                        epochs=NUM_EPOCHS,
                        steps_per_epoch=STEPS_PER_EPOCH,
                        validation_data=test_dataset,
                        validation_steps=VALIDATION_STEPS,
                        )

    return model, history
    

if __name__ == '__main__':
    ## Parameters
    TOPN = 1000

    RANDOM_STATE = 42
    tf.random.set_seed(RANDOM_STATE)

    BATCH_SIZE_LIST = [4, 8, 16, 32, 64]
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

    ## Model training
    print('============================================================')
    print('Model training')

    results = []
    for BATCH_SIZE, DROPOUT, LEARNING_RATE in itertools.product(BATCH_SIZE_LIST, DROPOUT_LIST, LEARNING_RATE_LIST):
        fdir_model = os.path.sep.join((newspath.fdir_model, f'TOPN-{TOPN}/dnn_regression/B-{BATCH_SIZE}_O-{DROPOUT}/'))
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

            train_dataset, test_dataset = build_dataset(x_train, x_test, y_train, y_test, BATCH_SIZE)
            INPUT_SHAPE = (x_train.shape[1],)
            STEPS_PER_EPOCH = x_train.shape[0]//BATCH_SIZE
            VALIDATION_STEPS = int(np.ceil(x_test.shape[0]/BATCH_SIZE))

            model = model_identification(INPUT_SHAPE, LEARNING_RATE)
            history = model.fit(train_dataset, epochs=NUM_EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, validation_data=test_dataset, validation_steps=VALIDATION_STEPS)

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
        print(f'BATCH_SIZE   : {BATCH_SIZE}')
        print(f'DROPOUT      : {DROPOUT}')
        print(f'LEARNING_RATE: {LEARNING_RATE:.,}')
        print('--------------------------------------------------')
        print('Model performance')
        print(f'  | MSE : {MSE:.03f}')
        print(f'  | MAPE: {MAPE:.03f}')

    ## Evaluation
    print('============================================================')
    print('Evaluation')

    for fpath_model, MSE, MAPE in sorted(results, key=lambda x:x[2], reverse=False):
        print(f'  | {fpath_model} -> MSE: {MSE:.03f} & MAPE: {MAPE:.03f}')