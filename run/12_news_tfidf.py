#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
rootpath = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1])
sys.path.append(rootpath)

from object import Word, NewsCorpus
from newsutil import NewsIO, NewsPath
newsio = NewsIO()
newspath = NewsPath()

import numpy as np
from tqdm import tqdm
from collections import defaultdict


def build_counter(do_build_counter, **kwargs):
    global fname_word_counter, fname_doc_counter

    if do_build_counter:
        corpus = kwargs.get('corpus', '')

        word_counter = defaultdict(int)
        doc_counter = defaultdict(int)
        for doc in corpus.iter():
            for w in doc.nouns_stop:
                word_counter[w] += 1

            for w in set(doc.nouns_stop):
                doc_counter[w] += 1

        newsio.save(_object=word_counter, _type='model', fname_object=fname_word_counter, verbose=False)
        newsio.save(_object=doc_counter, _type='model', fname_object=fname_doc_counter, verbose=False)

        print(f'  | fdir : {newspath.fdir_model}')
        print(f'  | fname_word_counter: {fname_word_counter}')
        print(f'  | fname_doc_counter: {fname_doc_counter}')

    else:
        word_counter = newsio.load(fname_object=fname_word_counter, _type='model', verbose=False)
        doc_counter = newsio.load(fname_object=fname_doc_counter, _type='model', verbose=False)

    return word_counter, doc_counter

def build_tfidf(do_build_tfidf, **kwargs):
    global fname_tfidf

    if do_build_tfidf:
        corpus = kwargs.get('corpus', '')
        word_counter = kwargs.get('word_counter', '')
        doc_counter = kwargs.get('doc_counter', '')

        tfidf = {}
        for w in tqdm(word_counter.keys()):
            word = Word(word=w)
            word.tf = word_counter[w]
            word.df = doc_counter[w]
            word.idf = np.log(DOCN / (word.df+1))
            word.tfidf = word.tf * word.idf

            tfidf[w] = word

        newsio.save(_object=word_counter, _type='model', fname_object=fname_word_counter, verbose=False)

    else:
        tfidf = newsio.load(fname_object=fname_tfidf, _type='model', verbose=False)

    return tfidf


if __name__ == '__main__':
    ## Filenames
    fname_tfidf = 'tfidf.pk'
    fname_word_counter = 'word_counter.pk'
    fname_doc_counter = 'doc_counter.pk'

    ## Parameters
    do_build_counter = True
    do_build_tfidf = True

    ## Data import
    print('============================================================')
    print('--------------------------------------------------')
    print('Load corpus')

    corpus = NewsCorpus(fdir_corpus=newspath.fdir_corpus)
    DOCN = len(corpus)
    print(f'  | Corpus: {DOCN:,}')

    ## Process
    print('============================================================')
    print('--------------------------------------------------')
    print('Frequency counter')

    word_counter, doc_counter = build_counter(corpus=corpus, do_build_counter=do_build_counter)
    print(f'  | {len(word_counter)} words in word_counter')

    print('--------------------------------------------------')
    print('TF-IDF')

    tfidf = build_tfidf(corpus=corpus, word_counter=word_counter, doc_counter=doc_counter, do_build_tfidf=do_build_tfidf)
    print(f'  | {len(tfidf)} words in tfidf')    

    print('  | Word     TF     DF     IDF     TF-IDF')
    for w, word in sorted(tfidf.items())[:10]:
        print('  | {:}  //  {:}  //  {:}  //  {:,.02f}  //  {:,.02f}'.format(word.word, word.tf, word.df, word.idf, word.tfidf))