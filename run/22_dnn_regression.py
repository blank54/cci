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

def model_identification(INPUT_SHAPE, ACTIVATION, LEARNING_RATE, LOSS):
    model = keras.Sequential()
    model.add(Dense(units=16, activation=ACTIVATION, input_shape=INPUT_SHAPE))
    model.add(Dropout(0.3))
    model.add(Dense(units=8, activation=ACTIVATION))
    model.add(Dropout(0.3))
    model.add(Dense(units=4, activation=ACTIVATION))
    model.add(Dropout(0.3))
    model.add(Dense(units=2, activation=ACTIVATION))
    model.add(Dropout(0.3))
    model.add(Dense(units=1))

    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss=LOSS)

    return model

def train(model, train_dataset, test_dataset, NUM_EPOCHS, STEPS_PER_EPOCH, VALIDATION_STEPS):
    history = model.fit(train_dataset,
                        epochs=NUM_EPOCHS,
                        steps_per_epoch=STEPS_PER_EPOCH,
                        validation_data=test_dataset,
                        validation_steps=VALIDATION_STEPS,
                        # callbacks=[EarlyStopping(monitor='val_loss', mode='min', patience=50)],
                        )

    return model, history
    


if __name__ == '__main__':
    ## Parameters
    TOPN = 1000
    RANDOM_STATE = 42

    tf.random.set_seed(RANDOM_STATE)
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 20000
    BATCH_SIZE = 32
    ACTIVATION = 'relu'
    LOSS = 'mse'

    DO_TRAIN = True

    ## Filenames
    fname_data_norm = f'data_w-{TOPN}_norm.pk'
    fname_correlated_variables = 'correlated_variables.json'

    fdir_model = os.path.sep.join((newspath.fdir_model, 'dnn_regression/D-16-8-4-2-1_O-3/'))
    os.makedirs(fdir_model, exist_ok=True)

    # fname_history = f'history_TOPN-{TOPN}_L-{LEARNING_RATE}_E-{NUM_EPOCHS}_B-{BATCH_SIZE}.pk'
    # fname_model = f'model_TOPN-{TOPN}_L-{LEARNING_RATE}_E-{NUM_EPOCHS}_B-{BATCH_SIZE}.pk'

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
    if DO_TRAIN:
        for LEARNING_RATE in [1e-3, 3e-4, 1e-4, 3e-5]:
            for BATCH_SIZE in [4, 8, 16, 32, 64]:
                fname_model = f'model_TOPN-{TOPN}_L-{LEARNING_RATE}_E-{NUM_EPOCHS}_B-{BATCH_SIZE}.pk'
                fname_history = f'history_TOPN-{TOPN}_L-{LEARNING_RATE}_E-{NUM_EPOCHS}_B-{BATCH_SIZE}.pk'
                
                train_dataset, test_dataset = build_dataset(x_train, x_test, y_train, y_test, BATCH_SIZE)
                INPUT_SHAPE = (x_train.shape[1],)
                STEPS_PER_EPOCH = x_train.shape[0]//BATCH_SIZE
                VALIDATION_STEPS = int(np.ceil(x_test.shape[0]/BATCH_SIZE))

                model = model_identification(INPUT_SHAPE, ACTIVATION, LEARNING_RATE, LOSS)
                model, history = train(model, train_dataset, test_dataset, NUM_EPOCHS, STEPS_PER_EPOCH, VALIDATION_STEPS)

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

    # RMSE = mean_squared_error(y_test, model.predict(x_test))**0.5
    # print(f'  | RMSE: {RMSE:,.03f}')

    for fname_model, RMSE in sorted(results, key=lambda x:x[1], reverse=False):
        print(f'{fname_model} -> {RMSE:.03f}')