#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
rootpath = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1])
sys.path.append(rootpath)

from object import NewsDate
from newsutil import NewsPath, NewsIO
newspath = NewsPath()
newsio = NewsIO()

import json
import random
import pickle as pk
from tqdm import tqdm
from copy import deepcopy
from datetime import datetime
random.seed(42)


def save_corpus(article):
    fpath_corpus = os.path.sep.join((newspath.fdir_corpus, article.fname))
    with open(fpath_corpus, 'wb') as g:
        pk.dump(article, g)

def save_corpus_monthly(article):
    yearmonth = datetime.strptime(article.date.datetime, '%Y.%m.%d').strftime('%Y%m')
    fpath_corpus_monthly = os.path.sep.join((newspath.fdir_corpus_monthly, yearmonth, article.fname))

    try:
        os.makedirs(os.path.dirname(fpath_corpus_monthly), exist_ok=False)
    except FileExistsError:
        pass

    with open(fpath_corpus_monthly, 'wb') as g:
        pk.dump(article, g)

def build_corpus(**kwargs):
    print('--------------------------------------------------')
    print('Find articles')

    fpath_article_list = []
    for path, dirs, files in os.walk(newspath.fdir_corpus):
        for fname in files:
            fpath_article = os.path.sep.join((path, fname))
            fpath_article_list.append(fpath_article)

    print(f'  | Articles: {len(fpath_article_list):,}')

    print('--------------------------------------------------')
    print('Articles -> Corpus')

    cnt = 0
    errors = []
    for fpath_article in tqdm(fpath_article_list):
        with open(fpath_article, 'rb') as f:
            article = pk.load(f)

            article.fname = deepcopy(os.path.basename(fpath_article))
            article.date = deepcopy(NewsDate(date=article.date))

            save_corpus(article=article)
            save_corpus_monthly(article=article)
            cnt += 1

    print(f'  | Corpus: {cnt:,}')
    print(f'  | Errors: {len(errors):,}')
    return errors


if __name__ == '__main__':
    ## Main
    print('============================================================')
    print('Build corpus')

    errors = build_corpus()

    ## Save errors
    if errors:
        print('============================================================')
        print('Save errors')

        fname_errors = 'corpus_errors.json'
        fpath_errors = os.path.sep.join((newspath.fdir_data, fname_errors))
        with open(fpath_errors, 'wb') as f:
            json.dump(errors, f)

        print(f'  | fdir : {newspath.fdir_data}')
        print(f'  | fname: {fname_errors}')
    else:
        pass