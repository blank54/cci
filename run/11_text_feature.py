#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
rootpath = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1])
sys.path.append(rootpath)

from news import NewsCorpus, NewsIO, NewsPath
newsio = NewsIO()
newspath = NewsPath()

import json
import itertools
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter


def build_counter(corpus):
    word_counter = defaultdict(int)
    doc_counter = defaultdict(int)
    attribute_errors = []
    for doc in corpus.iter():
        try:
            for w in itertools.chain(*doc['nouns_stop']):
                word_counter[w] += 1
            for w in set(itertools.chain(*doc['nouns_stop'])):
                doc_counter[w] += 1
        except AttributeError:
            attribute_errors.append(doc)

    if attribute_errors:
        print(f'  | errors during counting words: {len(attribute_errors):,}')
    else:
        pass

    return word_counter, doc_counter

def build_tfidf(corpus):
    DOCN = len(corpus)
    word_counter, doc_counter = build_counter(corpus)

    tfidf = defaultdict(dict)
    for w in tqdm(word_counter.keys()):
        word_tf = word_counter[w]
        word_df = doc_counter[w]
        word_idf = np.log(DOCN / (word_df+1))
        word_tfidf = word_tf*word_idf

        tfidf[w] = {
            'tf': word_tf,
            'df': word_df,
            'idf': word_idf,
            'tfidf': word_tfidf
        }

    return word_counter, tfidf
                                             
def word_count_diff(counter_now, counter_before):
    counter_diff = {}
    for word, count_now in counter_now.items():
        try:
            count_before = counter_before[word]
        except KeyError:
            count_before = 0

        counter_diff[word] = count_now - count_before
    return counter_diff

def word_count_ratio(counter_now, counter_before):
    counter_ratio = {}
    for word, count_now in counter_now.items():
        try:
            count_before = counter_before[word]
        except KeyError:
            count_before = 0.0001

        counter_ratio[word] = count_now / count_before
    return counter_ratio

def build_text_feature(corpus, ):
    print('## Feature: count')
    text_feature_first = defaultdict(dict)
    for yearmonth, docs in corpus.iter_month():
        word_list = []
        for doc in docs:
            for sent in doc['nouns_stop']:
                for word in sent:
                    word_list.append(word)

        text_feature_first[yearmonth]['doc_count'] = len(docs)
        text_feature_first[yearmonth]['word_count'] = Counter(word_list)

    print('## Feature: word count portion')
    yearmonth_list = sorted(corpus.yearmonth_list, reverse=False)
    for yearmonth in tqdm(yearmonth_list):
        text_feature_first[yearmonth]['word_count_portion'] = {w: c/sum(text_feature_first[yearmonth]['word_count'].values()) for w, c in text_feature_first[yearmonth]['word_count'].items()}

    print('## Feature: count - diff and ratio')
    text_feature_second = defaultdict(dict)
    for idx in tqdm(range(1, len(yearmonth_list))):
        _before = yearmonth_list[idx-1]
        _now = yearmonth_list[idx]

        text_feature_second[_now]['doc_count_diff'] = text_feature_first[_now]['doc_count'] - text_feature_first[_before]['doc_count']
        text_feature_second[_now]['doc_count_ratio'] = text_feature_first[_now]['doc_count'] / text_feature_first[_before]['doc_count']
        text_feature_second[_now]['word_count_diff'] = word_count_diff(text_feature_first[_now]['word_count'], text_feature_first[_before]['word_count'])
        text_feature_second[_now]['word_count_ratio'] = word_count_ratio(text_feature_first[_now]['word_count'], text_feature_first[_before]['word_count'])
        text_feature_second[_now]['word_count_portion_diff'] = word_count_diff(text_feature_first[_now]['word_count_portion'], text_feature_first[_before]['word_count_portion'])
        text_feature_second[_now]['word_count_portion_ratio'] = word_count_ratio(text_feature_first[_now]['word_count_portion'], text_feature_first[_before]['word_count_portion'])

    return text_feature_first, text_feature_second

def build_text_feature_topic(corpus):
    print('## Feature: topic count')
    text_feature_topic_first = defaultdict(dict)
    for yearmonth, docs in corpus.iter_month():
        if yearmonth in text_feature_topic_first.keys():
            pass
        else:
            text_feature_topic_first[yearmonth]['count'] = defaultdict(int)

        for doc in docs:
            topic_id = doc['topic_id']
            text_feature_topic_first[yearmonth]['count'][topic_id] += 1

    print('## Feature: topic portion')
    yearmonth_list = sorted(corpus.yearmonth_list, reverse=False)
    for yearmonth in tqdm(yearmonth_list):
        topic_list = list(text_feature_topic_first[yearmonth]['count'].keys())
        text_feature_topic_first[yearmonth]['portion'] = {}
        topic_sum = sum(text_feature_topic_first[yearmonth]['count'].values())
        for topic_id in topic_list:
            text_feature_topic_first[yearmonth]['portion'][topic_id] = text_feature_topic_first[yearmonth]['count'][topic_id] / topic_sum

    print('## Feature: topic count - diff and ratio')
    text_feature_topic_second = defaultdict(dict)
    for idx in tqdm(range(1, len(yearmonth_list))):
        _before = yearmonth_list[idx-1]
        _now = yearmonth_list[idx]

        topic_list = list(text_feature_topic_first[yearmonth]['count'].keys())
        text_feature_topic_second[_now]['diff'] = defaultdict(float)
        text_feature_topic_second[_now]['ratio'] = defaultdict(float)
        for topic_id in topic_list:
            text_feature_topic_second[_now]['diff'][topic_id] = text_feature_topic_first[_now]['count'][topic_id] - text_feature_topic_first[_before]['count'][topic_id]
            try:
                text_feature_topic_second[_now]['ratio'][topic_id] = text_feature_topic_first[_now]['count'][topic_id] / text_feature_topic_first[_before]['count'][topic_id]
            except ZeroDivisionError:
                text_feature_topic_second[_now]['ratio'][topic_id] = text_feature_topic_first[_now]['count'][topic_id] / 1e4

    return text_feature_topic_first, text_feature_topic_second


if __name__ == '__main__':
    ## Filenames
    fname_text_feature_first = 'text_feature_first.json'
    fname_text_feature_second = 'text_feature_second.json'
    fname_text_feature_topic_first = 'text_feature_topic_first.json'
    fname_text_feature_topic_second = 'text_feature_topic_second.json'

    fname_tfidf = 'tfidf.pk'
    fname_word_counter = 'word_counter.pk'
    fname_doc_counter = 'doc_counter.pk'

    fname_target_word_list = 'target_word_list.json'

    ## Parameters
    DO_WORD = False
    DO_TOPIC = False
    DO_BUILD_TFIDF = False
    TOPN = 1000

    ## Data import
    print('============================================================')
    print('Load corpus')

    corpus = NewsCorpus(dname_corpus='corpus_topic_filtered',
                        start='200501',
                        end='201912')

    ## Process
    print('============================================================')
    print('Text feature engineering')

    if DO_WORD:
        text_feature_first, text_feature_second = build_text_feature(corpus)
        newsio.save_json(_object=text_feature_first, _type='data', fname_object=fname_text_feature_first, verbose=False)
        newsio.save_json(_object=text_feature_second, _type='data', fname_object=fname_text_feature_second, verbose=False)
    else:
        text_feature_first = newsio.load_json(fname_object=fname_text_feature_first, _type='data')
        text_feature_second = newsio.load_json(fname_object=fname_text_feature_second, _type='data')


    if DO_TOPIC:
        text_feature_topic_first, text_feature_topic_second = build_text_feature_topic(corpus)
        newsio.save_json(_object=text_feature_topic_first, _type='data', fname_object=fname_text_feature_topic_first, verbose=False)
        newsio.save_json(_object=text_feature_topic_second, _type='data', fname_object=fname_text_feature_topic_second, verbose=False)
    else:
        text_feature_topic_first = newsio.load_json(fname_object=fname_text_feature_topic_first, _type='data')
        text_feature_topic_second = newsio.load_json(fname_object=fname_text_feature_topic_second, _type='data')

    print('  | Evaluation of text features')
    word_set = []
    for yearmonth in tqdm(corpus.yearmonth_list):
        word_set.extend(text_feature_first[yearmonth]['word_count'].values())
    print('  | # of individual words (Feature engineering): {:,}'.format(len(word_set)))

    print('============================================================')
    print('Build TF-IDF')

    if DO_BUILD_TFIDF:
        word_counter, tfidf = build_tfidf(corpus=corpus)
        newsio.save(_object=word_counter, _type='model', fname_object=fname_word_counter, verbose=False)
        newsio.save(_object=doc_counter, _type='model', fname_object=fname_doc_counter, verbose=False)
        newsio.save(_object=tfidf, _type='model', fname_object=fname_tfidf, verbose=False)
    else:
        word_counter = newsio.load(fname_object=fname_word_counter, _type='model')
        doc_counter = newsio.load(fname_object=fname_doc_counter, _type='model')
        tfidf = newsio.load(fname_object=fname_tfidf, _type='model')

    print('  | # of individual words (TF-IDF): {:,}'.format(len(word_counter)))

    print('============================================================')
    print('Select TOPN words')

    word_sorted = list(sorted([(word, tfidf[word]['tfidf']) for word in tfidf], key=lambda x:x[1], reverse=True))[:TOPN]
    
    print(f'  | # of words: {len(word_sorted)}')
    for word, _ in word_sorted[:30]:
        print('  | {}: {:.02f} -> {:.02f} -> {:.02f}'.format(word, tfidf[word]['tf'], tfidf[word]['df'], tfidf[word]['tfidf']))

    newsio.save_json(_object=word_sorted, _type='data', fname_object=fname_target_word_list)