#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
rootpath = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1])
sys.path.append(rootpath)

from news import NewsIO, NewsPath
newsio = NewsIO()
newspath = NewsPath()

import pandas as pd
from tqdm import tqdm
from collections import defaultdict


def build_numeric_feature(numeric_data):
    variable_list = [var for var in numeric_data.keys() if var not in ['Unnamed: 0', 'yearmonth']]

    print('## Feature: raw')
    numeric_feature_first = defaultdict(dict)
    for yearmonth in tqdm(numeric_data['yearmonth'].tolist()):
        numeric_feature_first[yearmonth]['raw'] = {}
        for var in variable_list:
            numeric_feature_first[yearmonth]['raw'][var] = float(numeric_data.loc[numeric_data['yearmonth']==yearmonth][var])

    print('## Feature: diff and ratio')
    numeric_feature_second = defaultdict(dict)
    for idx in tqdm(range(1, len(numeric_data))):
        row_before = numeric_data.iloc[idx-1]
        row_now = numeric_data.iloc[idx]
        
        numeric_feature_second[row_now['yearmonth']]['diff'] = {}
        numeric_feature_second[row_now['yearmonth']]['ratio'] = {}
        for var in variable_list:
            diff_value = numeric_feature_first[row_now['yearmonth']]['raw'][var] - numeric_feature_first[row_before['yearmonth']]['raw'][var]
            try:
                ratio_value = numeric_feature_first[row_now['yearmonth']]['raw'][var] / numeric_feature_first[row_before['yearmonth']]['raw'][var]
            except ZeroDivisionError:
                ratio_value = numeric_feature_first[row_now['yearmonth']]['raw'][var] / 1e4

            numeric_feature_second[row_now['yearmonth']]['diff'][var] = diff_value
            numeric_feature_second[row_now['yearmonth']]['ratio'][var] = ratio_value

    return numeric_feature_first, numeric_feature_second




if __name__ == '__main__':
    ## Filenames
    fname_numeric_data_norm = 'numeric_data_norm.xlsx'
    fpath_numeric_data_norm = os.path.sep.join((newspath.fdir_data, fname_numeric_data_norm))

    fname_numeric_feature_first = 'numeric_feature_first.json'
    fname_numeric_feature_second = 'numeric_feature_second.json'

    ## Parameters
    DO_FEATURE_ENGINEERING = fname_numeric_feature_second

    ## Data import
    print('============================================================')
    print('Load numeric data')

    numeric_data = pd.read_excel(fpath_numeric_data_norm)
    numeric_data['yearmonth'] = [str(ym) for ym in numeric_data['yearmonth'].tolist()]
    print('  | Shape of dataset: {}'.format(numeric_data.shape))

    ## Process
    print('============================================================')
    print('Numeric feature engineering')

    if DO_FEATURE_ENGINEERING:
        numeric_feature_first, numeric_feature_second = build_numeric_feature(numeric_data)
        newsio.save_json(_object=numeric_feature_first, _type='data', fname_object=fname_numeric_feature_first)
        newsio.save_json(_object=numeric_feature_second, _type='data', fname_object=fname_numeric_feature_second)
    else:
        numeric_feature_first = newsio.load_json(fname_object=fname_numeric_feature_first, _type='data')
        numeric_feature_second = newsio.load_json(fname_object=fname_numeric_feature_second, _type='data')