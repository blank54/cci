#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
rootpath = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1])
sys.path.append(rootpath)

from object import LdaGridSearchResult
from newsutil import NewsIO, NewsPath
newsio = NewsIO()
newspath = NewsPath()

import numpy as np
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt


def find_optimum(lda_gs_result):
    result_list = [(fname, coherence) for fname, coherence in zip(lda_gs_result.result['fname'], lda_gs_result.result['coherence'])]
    result_list_sorted = list(sorted(result_list, key=lambda x:x[1], reverse=True))
    return result_list_sorted[0]


if __name__ == '__main__':
    ## Filenames
    SAMPLE_SIZE = 100000
    fname_gs_result = f'lda_gs_{SAMPLE_SIZE}.json'

    ## Parameters
    VIS_GS = False

    ## Data import
    print('============================================================')
    print('Grid search')

    gs_result = newsio.load(_type='result', fname_object=fname_gs_result)
    lda_gs_result = LdaGridSearchResult(gs_result=gs_result)

    ## Visualize grid search results
    if VIS_GS:
        print('--------------------------------------------------')
        print('num_topics')

        lda_gs_result.box_plot('num_topics')

        print('--------------------------------------------------')
        print('iterations')

        lda_gs_result.box_plot('iterations')

        print('--------------------------------------------------')
        print('alpha')

        lda_gs_result.box_plot('alpha')

        print('--------------------------------------------------')
        print('eta')

        lda_gs_result.box_plot('eta')
    else:
        pass

    ## Find optimum
    print('--------------------------------------------------')
    print('Optimum model')

    fname_lda_opt, coherence_opt = find_optimum(lda_gs_result)
    lda_model = newsio.load(_type='model', fname_object=fname_lda_opt)
    print(f'  | fname    : {fname_lda_opt}')
    print(f'  | model    : {type(lda_model)}')
    print(f'  | coherence: {coherence_opt:,.03f}')