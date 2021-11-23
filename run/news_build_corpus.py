#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
rootpath = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1])
sys.path.append(rootpath)

from newsutil import NewsPath, NewsIO, NewsFunc
newspath = NewsPath()
newsio = NewsIO()
newsfunc = NewsFunc()

import numpy as np
import pickle as pk
from collections import defaultdict


def build_corpus(fdir):
    total = len(os.listdir(fdir))
    cnt = 0

    corpus = defaultdict(dict)
    for path, dirs, files in os.walk(fdir):
        for f in files:
            fpath = os.path.sep.join((path, f))
            with open(fpath, 'rb') as f:
                article = pk.load(f)

            corpus[article.id] = {'url': article.url,
                                  'query': '  '.join(article.query),
                                  'title': article.title,
                                  'date': article.date,
                                  'category': article.category,
                                  'content': article.content,
                                  }

            cnt += 1
            sys.stdout.write('\r  | {:,} articles ({:,.02f} % | total: {:,})'.format(cnt, ((cnt*100)/total), total))

    print()
    return corpus


if __name__ == '__main__':
    ## Filenames
    fname_corpus = 'corpus_{}.json'.format(str(sys.argv[1]))

    ## Build corpus
    print('============================================================')
    print('Build corpus')

    fdir = newspath.fdir_article
    corpus = build_corpus(fdir=fdir)

    print('  | # of articles: {:,}'.format(len(corpus)))

    ## Save corpus
    print('============================================================')
    print('Save corpus')

    fpath_corpus = os.path.join(newspath.fdir_corpus, fname_corpus)
    newsio.save_corpus(corpus=corpus, fpath=fpath_corpus)

    print('  | fdir : {}'.format(newspath.fdir_corpus))
    print('  | fname: {}'.format(fname_corpus))