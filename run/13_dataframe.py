#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
rootpath = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1])
sys.path.append(rootpath)

from news import NewsIO, NewsFunc
newsio = NewsIO()
newsfunc = NewsFunc()

import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict


def load_features(fname_feature_dict):
    feature_dict = {
        'target_word_list': [w for w, v in newsio.load_json(fname_object=fname_feature_dict['fname_target_word_list'], _type='data')][:TOPN],

        'text_feature_first': newsio.load_json(fname_object=fname_feature_dict['fname_text_feature_first'], _type='data'),
        'text_feature_second': newsio.load_json(fname_object=fname_feature_dict['fname_text_feature_second'], _type='data'),
        'text_feature_topic_first': newsio.load_json(fname_object=fname_feature_dict['fname_text_feature_topic_first'], _type='data'),
        'text_feature_topic_second': newsio.load_json(fname_object=fname_feature_dict['fname_text_feature_topic_second'], _type='data'),

        'numeric_feature_first': newsio.load_json(fname_object=fname_feature_dict['fname_numeric_feature_first'], _type='data'),
        'numeric_feature_second': newsio.load_json(fname_object=fname_feature_dict['fname_numeric_feature_second'], _type='data'),
    }
    return feature_dict
    
def build_data(feature_dict):
    yearmonth_list = list(feature_dict['text_feature_first'].keys())
    topic_list = list(feature_dict['text_feature_topic_first']['200501']['count'].keys())
    numeric_variable_list = list(feature_dict['numeric_feature_first']['200501']['raw'].keys())

    data = defaultdict(list)
    for yearmonth in tqdm(yearmonth_list[1:]):
        data['yearmonth'].append(yearmonth)
        
        # for word in feature_dict['target_word_list']:
        #     try:
        #         data[f'word_count_{word}'].append(feature_dict['text_feature_first'][yearmonth]['word_count'][word])
        #     except KeyError:
        #         data[f'word_count_{word}'].append(0)
                
        #     try:
        #         data[f'word_count_portion_{word}'].append(feature_dict['text_feature_first'][yearmonth]['word_count_portion'][word])
        #     except KeyError:
        #         data[f'word_count_portion_{word}'].append(0)
                
        #     try:
        #         data[f'word_count_diff_{word}'].append(feature_dict['text_feature_second'][yearmonth]['word_count_diff'][word])
        #     except KeyError:
        #         data[f'word_count_diff_{word}'].append(0)
                
        #     try:
        #         data[f'word_count_ratio_{word}'].append(feature_dict['text_feature_second'][yearmonth]['word_count_ratio'][word])
        #     except KeyError:
        #         data[f'word_count_ratio_{word}'].append(0)
                
        #     try:
        #         data[f'word_count_portion_diff_{word}'].append(feature_dict['text_feature_second'][yearmonth]['word_count_portion_diff'][word])
        #     except KeyError:
        #         data[f'word_count_portion_diff_{word}'].append(0)
                
        #     try:
        #         data[f'word_count_portion_ratio_{word}'].append(feature_dict['text_feature_second'][yearmonth]['word_count_portion_ratio'][word])
        #     except KeyError:
        #         data[f'word_count_portion_ratio_{word}'].append(0)
                
        # data['doc_count'].append(feature_dict['text_feature_first'][yearmonth]['doc_count'])
        # data['doc_count_diff'].append(feature_dict['text_feature_second'][yearmonth]['doc_count_diff'])
        # data['doc_count_ratio'].append(feature_dict['text_feature_second'][yearmonth]['doc_count_ratio'])
            
        # for topic_id in topic_list:
        #     try:
        #         data[f'topic_count_{topic_id}'].append(feature_dict['text_feature_topic_first'][yearmonth]['count'][topic_id])
        #     except KeyError:
        #         data[f'topic_count_{topic_id}'].append(0)
                
        #     try:
        #         data[f'topic_count_portion_{topic_id}'].append(feature_dict['text_feature_topic_first'][yearmonth]['portion'][topic_id])
        #     except KeyError:
        #         data[f'topic_count_portion_{topic_id}'].append(0)
            
        #     data[f'topic_count_diff_{topic_id}'].append(feature_dict['text_feature_topic_second'][yearmonth]['diff'][topic_id])
        #     data[f'topic_count_ratio_{topic_id}'].append(feature_dict['text_feature_topic_second'][yearmonth]['ratio'][topic_id])
            
        for var in numeric_variable_list:
            data[f'numeric_raw_{var}'].append(feature_dict['numeric_feature_first'][yearmonth]['raw'][var])
            data[f'numeric_diff_{var}'].append(feature_dict['numeric_feature_second'][yearmonth]['diff'][var])
            data[f'numeric_ratio_{var}'].append(feature_dict['numeric_feature_second'][yearmonth]['ratio'][var])
            
    return pd.DataFrame(data)

def normalize_data(df):
    except_list = ['yearmonth']
    demographic_info_df = newsfunc.explore_demographic_info(df=df, except_list=except_list)
    variable_list_meanstd, variable_list_minmax = newsfunc.partition_variable_list(demographic_info_df)

    print('## Mean-standard normalization')
    print('  | Variables: {:,}'.format(len(variable_list_meanstd)))
    df_meanstd = deepcopy(df[variable_list_meanstd])
    df_meanstd_norm = newsfunc.normalize_meanstd(df_meanstd)
    print('  | data shape: {}'.format(df_meanstd_norm.shape))
    df_meanstd_norm = deepcopy(pd.concat([df_meanstd_norm, df[except_list]], axis=1))

    print('## Min-max normalization')
    print('  | Variables: {:,}'.format(len(variable_list_minmax)))
    df_minmax = deepcopy(df[variable_list_minmax])
    df_minmax_norm = newsfunc.normalize_minmax(df_minmax)
    print('  | data shape: {}'.format(df_minmax_norm.shape))
    df_minmax_norm = deepcopy(pd.concat([df_minmax_norm, df[except_list]], axis=1))

    print('## Merge normalized data')
    df_norm = deepcopy(pd.merge(df_meanstd_norm, df_minmax_norm))
    print('  | data shape: {}'.format(df_norm.shape))

    return df_norm


if __name__ == '__main__':
    ## Parameters
    TOPN = 1000

    DO_BUILD_DATA = True
    DO_NORMALIZE = True

    ## Filenames
    fname_feature_dict = {
        'fname_target_word_list': 'target_word_list.json',

        'fname_text_feature_first': 'text_feature_first.json',
        'fname_text_feature_second': 'text_feature_second.json',
        'fname_text_feature_topic_first': 'text_feature_topic_first.json',
        'fname_text_feature_topic_second': 'text_feature_topic_second.json',
        
        'fname_numeric_feature_first': 'numeric_feature_first.json',
        'fname_numeric_feature_second': 'numeric_feature_second.json',
    }

    fname_data = f'data_w-{TOPN}.pk'
    fname_data_norm = f'data_w-{TOPN}_norm.pk'

    ## Data import
    print('============================================================')
    print('Load features')
    feature_dict = load_features(fname_feature_dict)

    ## Build dataframe
    print('============================================================')
    print('Build data')

    if DO_BUILD_DATA:
        df = build_data(feature_dict)
        newsio.save(_object=df, _type='data', fname_object=fname_data)
    else:
        df = newsio.load(fname_object=fname_data, _type='data')

    print('  | Data shape: {}'.format(df.shape))

    ## Normalization
    print('============================================================')
    print('Normalization')

    if DO_NORMALIZE:
        df_norm = normalize_data(df)
        newsio.save(_object=df_norm, _type='data', fname_object=fname_data_norm)
    else:
        df_norm = newsio.load(fname_object=fname_data_norm, _type='data')

    print('  | Data shape: {}'.format(df_norm.shape))