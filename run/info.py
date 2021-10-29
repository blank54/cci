#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
rootpath = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1])
sys.path.append(rootpath)

from newsutil import NewsPath, NewsIO
newspath = NewsPath()
newsio = NewsIO()

import pickle as pk


if __name__ == '__main__':
    print('============================================================')
    print('Articles')

    flist_article = os.listdir(newspath.fdir_article)

    print('  | fdir : {}'.format(newspath.fdir_article))
    print('  | # of articles: {:,}'.format(len(flist_article)))

    print('============================================================')
    print('Corpus')

    fname_corpus = list(sorted(os.listdir(newspath.fdir_corpus), reverse=True))[0]
    fpath_corpus = os.path.sep.join((newspath.fdir_corpus, fname_corpus))
    with open(fpath_corpus, 'rb') as f:
        corpus = pk.load(f)

    print('  | fdir: {}'.format(newspath.fdir_corpus))
    print('  | fname(most recent): {}'.format(fname_corpus))
    print('  | # of articles in corpus: {:,}'.format(len(corpus)))