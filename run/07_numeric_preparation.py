#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
rootpath = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1])
sys.path.append(rootpath)

from news import NewsPath, NewsIO, NewsFunc, NumericData
newspath = NewsPath()
newsio = NewsIO()
newsfunc = NewsFunc()

import pandas as pd
from copy import deepcopy
from collections import defaultdict


if __name__ == '__main__':
    ## Filenames
    fdir_numeric_data = os.path.sep.join((newspath.fdir_data, 'numeric'))

    fname_demographic_info = 'numeric_demographic_info.xlsx'
    fpath_demographic_info = os.path.sep.join((newspath.fdir_data, fname_demographic_info))

    fname_demographic_info_norm = 'numeric_demographic_info_norm.xlsx'
    fpath_demographic_info_norm = os.path.sep.join((newspath.fdir_data, fname_demographic_info_norm))
    
    fname_numeric_data_norm = 'numeric_data_norm.xlsx'
    fpath_numeric_data_norm = os.path.sep.join((newspath.fdir_data, fname_numeric_data_norm))


    ## Data import
    print('============================================================')
    print('Data import')

    numeric_data = NumericData(fdir=fdir_numeric_data, start='200501', end='201912')
    numeric_df = numeric_data.to_df()
    print('  | Shape of dataset: {}'.format(numeric_df.shape))

    ## Demographic information
    demographic_info_df = newsfunc.explore_demographic_info(numeric_df, except_list=['yearmonth'])
    demographic_info_df.to_excel(excel_writer=fpath_demographic_info)

    ## Normalization
    print('============================================================')
    print('Partition variable list')
    variable_list_meanstd, variable_list_minmax = newsfunc.partition_variable_list(demographic_info_df)

    print('  | Mean-std ({:,}): {}, ...'.format(len(variable_list_meanstd), ', '.join(variable_list_meanstd[:5])))
    print('  | Min-max  ({:,}): {}, ...'.format(len(variable_list_minmax), ', '.join(variable_list_minmax[:5])))

    print('Normalization')

    df_meanstd = deepcopy(numeric_df[variable_list_meanstd])
    df_meanstd_norm = newsfunc.normalize_meanstd(df_meanstd)
    df_meanstd_norm = deepcopy(pd.concat([df_meanstd_norm, numeric_df['yearmonth']], axis=1))

    df_minmax = deepcopy(numeric_df[variable_list_minmax])
    df_minmax_norm = newsfunc.normalize_minmax(df_minmax)
    df_minmax_norm = deepcopy(pd.concat([df_minmax_norm, numeric_df['yearmonth']], axis=1))

    numeric_df_norm = deepcopy(pd.merge(df_meanstd_norm, df_minmax_norm))
    demographic_info_norm = newsfunc.explore_demographic_info(numeric_df_norm, except_list=['yearmonth'])

    demographic_info_norm.to_excel(excel_writer=fpath_demographic_info_norm)
    numeric_df_norm.to_excel(excel_writer=fpath_numeric_data_norm)

    print('  | Shape of dataset: {}'.format(numeric_df_norm.shape))