#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
rootpath = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1])
sys.path.append(rootpath)

from news import NewsCorpus, LdaGridSearchResult, NewsIO, NewsPath
newsio = NewsIO()
newspath = NewsPath()

import json
import itertools
import numpy as np
from copy import deepcopy
from collections import defaultdict


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

def assign_topic_to_articles(fname_lda_opt, fname_id2word, corpus):
    lda_model = newsio.load(_type='model', fname_object=fname_lda_opt)
    id2word = newsio.load(_type='data', fname_object=fname_id2word)

    for doc in corpus.iter():
        ## Assign topic id
        doc_bow = id2word.doc2bow(itertools.chain(*doc['nouns_stop']))
        topic_scores = lda_model.inference([doc_bow])[0]
        topic_id = np.argmax(topic_scores)

        doc['fname_lda_model'] = fname_lda_opt
        doc['topic_id'] = str(topic_id)

        ## Save corpus
        fpath_original = doc['fpath_article_corpus']
        fpath_processed = fpath_original.replace('corpus', 'corpus_topic_assigned')
        doc['fpath_article_corpus'] = deepcopy(fpath_processed)
        os.makedirs(os.path.dirname(fpath_processed), exist_ok=True)

        with open(fpath_processed, 'w', encoding='utf-8') as f:
            json.dump(doc, f)

def filter_corpus_by_topic(corpus_filtered):
    for doc in corpus_filtered.iter():
        ## Save corpus
        fpath_original = doc['fpath_article_corpus']
        fpath_processed = fpath_original.replace('corpus_topic_assigned', 'corpus_topic_filtered')
        doc['fpath_article_corpus'] = deepcopy(fpath_processed)
        os.makedirs(os.path.dirname(fpath_processed), exist_ok=True)

        with open(fpath_processed, 'w', encoding='utf-8') as f:
            json.dump(doc, f)


if __name__ == '__main__':
    ## Filenames
    SAMPLE_SIZE = 100000
    fname_gs_result = f'lda_gs_{SAMPLE_SIZE}.json'
    fname_id2word = f'lda/id2word_{SAMPLE_SIZE}.json'

    fname_topic_keywords = 'topic_keywords.xlsx'

    ## Parameters
    MAX_WORD_TOPIC = 50

    VIS_GS = False
    TOPIC_NAME_ASSIGNMENT = False
    DOC_TOPIC_ASSIGNMENT = False
    TOPIC_FILTER = True

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
    print('============================================================')
    print('Optimum model')

    fname_lda_opt, coherence_opt = find_optimum(lda_gs_result)
    lda_model = newsio.load(_type='model', fname_object=fname_lda_opt)
    print(f'  | model    : {type(lda_model)}')
    print(f'  | coherence: {coherence_opt:,.03f}')

    ## Topic name assignment
    if TOPIC_NAME_ASSIGNMENT:
        export_topic_keywords(lda_model, MAX_WORD_TOPIC, fname_topic_keywords)
    else:
        pass

    ## Topic assignment
    print('============================================================')
    print('Topic assignment')
    print('--------------------------------------------------')
    print('Load corpus')

    corpus = NewsCorpus(start='200501', end='201912')
    DOCN = len(corpus)
    print(f'  | Corpus: {DOCN:,}')

    print('--------------------------------------------------')
    print('Assign topic for each doc')

    if DOC_TOPIC_ASSIGNMENT:
        assign_topic_to_articles(fname_lda_opt, fname_id2word, corpus)
    else:
        pass

    print('--------------------------------------------------')
    print('Filter articles by topics')

    topic_ids_filtered = ['6', '12', '16', '27']
    corpus_filtered = NewsCorpus(fdir_corpus=os.path.sep.join((newspath.root, 'corpus_topic_assigned')),
                                 topic_filtered=True,
                                 topic_ids=topic_ids_filtered,
                                 start='200501',
                                 end='201912')

    if TOPIC_FILTER:
        filter_corpus_by_topic(corpus_filtered)
    else:
        pass