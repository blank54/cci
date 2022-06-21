#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
rootpath = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1])
sys.path.append(rootpath)

from object import NewsCorpusMonthly, LdaGridSearchResult
from newsutil import NewsIO, NewsPath
newsio = NewsIO()
newspath = NewsPath()

import itertools
import numpy as np
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt


def show_grid_search_result(lda_gs_result):
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

def find_optimum(lda_gs_result):
    result_list = [(fname, coherence) for fname, coherence in zip(lda_gs_result.result['fname'], lda_gs_result.result['coherence'])]
    result_list_sorted = list(sorted(result_list, key=lambda x:x[1], reverse=True))
    return result_list_sorted[0]

def export_topic_keywords(lda_model, MAX_WORD_TOPIC, fname_topic_keywords):
    topics = lda_model.show_topics(num_topics=lda_model.num_topics, num_words=MAX_WORD_TOPIC, formatted=False)

    topic_keywords = defaultdict(list)
    for topic_id, word_list in topics:
        topic_keywords['topic_id'].append(topic_id)
        for idx, (word, score) in enumerate(word_list):
            topic_keywords[f'word_{idx}'].append(word)
            
    topic_keywords_df = pd.DataFrame(topic_keywords)
    topic_keywords_df.to_excel(excel_writer=os.path.sep.join((newspath.fdir_result, fname_topic_keywords)))

def assign_topic_to_articles():



if __name__ == '__main__':
    ## Filenames
    SAMPLE_SIZE = 100000
    fname_gs_result = f'lda_gs_{SAMPLE_SIZE}.json'

    fname_docs_dict = f'lda/docs_dict_{SAMPLE_SIZE}.json'
    fname_id2word = f'lda/id2word_{SAMPLE_SIZE}.json'
    fname_docs_bow = f'lda/docs_bow_{SAMPLE_SIZE}.json'

    fname_topic_keywords = 'topic_keywords.xlsx'

    ## Parameters
    MAX_WORD_TOPIC = 50

    VIS_GS = False
    TOPIC_NAME_ASSIGNMENT = False

    ## Data import
    print('============================================================')
    print('Grid search')

    gs_result = newsio.load(_type='result', fname_object=fname_gs_result)
    lda_gs_result = LdaGridSearchResult(gs_result=gs_result)

    ## Visualize grid search results
    if VIS_GS:
        show_grid_search_result(lda_gs_result=lda_gs_result)
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

    ## Topic name assignment
    docs_dict = newsio.load(_type='data', fname_object=fname_docs_dict)
    id2word = newsio.load(_type='data', fname_object=fname_id2word)
    docs_bow = newsio.load(_type='data', fname_object=fname_docs_bow)

    if TOPIC_NAME_ASSIGNMENT:
        export_topic_keywords(lda_model, MAX_WORD_TOPIC, fname_topic_keywords)
    else:
        pass

    ## Topic assignment