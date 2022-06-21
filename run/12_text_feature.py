#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
rootpath = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1])
sys.path.append(rootpath)

from object import NewsMonthlyCorpus
from newsutil import NewsIO, NewsPath
newsio = NewsIO()
newspath = NewsPath()

import itertools
import pickle as pk
from tqdm import tqdm
from collections import defaultdict


if __name__ == '__main__':
    ## Filenames
    fname_word_counter = 'monthly_word_counter.pk'

    ## Data import
    print('============================================================')
    print('--------------------------------------------------')
    print('Load corpus')

    corpus = NewsMonthlyCorpus(fdir_corpus=newspath.fdir_corpus_monthly, start='200501', end='201912')
    print(f'  | Corpus: {len(corpus):,}')

    word_counter = {}
    for yearmonth in tqdm(corpus.yearmonth_list):
        word_counter[yearmonth] = defaultdict(int)

        fdir = os.path.sep.join((newspath.fdir_corpus_monthly, yearmonth))
        for fname in os.listdir(fdir):
            fpath = os.path.sep.join((fdir, fname))

            with open(fpath, 'rb') as f:
                article = pk.load(f)

            for w in itertools.chain(*article.nouns_stop):
                word_counter[yearmonth][w] += 1

    newsio.save(_object=word_counter, _type='model', fname_object=fname_word_counter, verbose=True)