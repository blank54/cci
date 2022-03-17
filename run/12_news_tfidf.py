#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
rootpath = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1])
sys.path.append(rootpath)

from object import Word
from newsutil import NewsIO
newsio = NewsIO()

import numpy as np
from tqdm import tqdm
from collections import Counter


def calculate_tfidf(corpus, word_counter):
    docs_for_tfidf = sum(corpus.values(), [])
    NUM_DOC = len(docs_for_tfidf)

    tfidf = {}
    for w in tqdm(word_counter):
        word = Word(word=w)

        word.tf = word_counter[word]
        word.df = len([sent for sent in docs_for_tfidf if word in sent])
        word.idf = np.log(NUM_DOC / (word.df+1))
        word.tfidf = word.tf * word.idf

        tfidf[w] = word

    return tfidf


if __name__ == '__main__':
    ## Filenames
    fname_corpus = 'corpus_1000_noun.pk'
    fname_stoplist = 'stoplist.txt'

    fname_word_counter = 'word_counter_1000.pk'
    fname_tfidf = 'tfidf_1000.pk'

    ## Data import
    print('============================================================')
    print('Load corpus')

    corpus = newsio.load(fname_object=fname_corpus, _type='corpus')
    stoplist = newsio.read_thesaurus(fname_thesaurus=fname_stoplist)

    print(f'  | Corpus: {len(corpus):,}')
    print(f'  | Stopwords: {stoplist}')

    ## Term frequency
    print('============================================================')
    print('Term frequency')

    word_counter = Counter(sum(sum(corpus.values(), []), []))
    newsio.save(_object=word_counter, _type='model', fname_object=fname_word_counter)

    print(f'  | Terms: {len(word_counter):,}')

    ## TF-IDF
    print('============================================================')
    print('TF-IDF')

    tfidf = calculate_tfidf(corpus=corpus, word_counter=word_counter)
    newsio.save(_object=tfidf, _type='model', fname_object=fname_tfidf)

    print(f'  | TFIDFs: {len(tfidf):,}')