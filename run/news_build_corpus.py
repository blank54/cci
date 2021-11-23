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
    cnt_success = 0
    cnt_error = 0

    errors = []
    for path, dirs, files in os.walk(fdir):
        for fname in files:
            cnt += 1
            fpath_data = os.path.sep.join((path, fname))
            with open(fpath_data, 'rb') as fd:
                try:
                    article = pk.load(fd)

                    yearmonth = str(article.date[:6])
                    fdir_yearmonth = os.path.sep.join((newspath.fdir_corpus, 'Q-건설', yearmonth))
                    if not os.path.isdir(fdir_yearmonth):
                        os.makedirs(fdir_yearmonth)
                    else:
                        pass

                    fpath_corpus = os.path.sep.join((fdir_yearmonth, fname))
                    with open(fpath_corpus, 'wb') as fc:
                        pk.dump(article, fc)

                    cnt_success += 1

                except EOFError:
                    errors.append(fpath_data)
                    cnt_error += 1
            
            sys.stdout.write('\r  | success-{:,} / error-{:,} ({:,.02f} % from total {:,} articles)'.format(cnt_success, cnt_error, ((cnt*100)/total), total))

    print()
    return errors


if __name__ == '__main__':
    ## Build corpus
    print('============================================================')
    print('Build corpus')

    errors = build_corpus(fdir=newspath.fdir_article)

    print('  | fdir  : {}'.format(newspath.fdir_corpus))
    print('  | errors: {:,} articles'.format(len(errors)))