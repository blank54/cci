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

import json
import random
import pickle as pk
from tqdm import tqdm
random.seed(42)

def build_corpus(**kwargs):
    print('--------------------------------------------------')
    print('Find articles')

    flist = []
    for path, dirs, files in os.walk(newspath.fdir_article):
        for fname in files:
            flist.append(fname)

    print(f'  | Articles: {len(flist):,}')

    print('--------------------------------------------------')
    print('Articles -> Corpus')

    cnt = 0
    errors = []
    for fname in tqdm(flist):
        try:
            fpath_article = os.path.sep.join((path, fname))
            fpath_corpus = os.path.sep.join((newspath.fdir_corpus, fname))
            with open(fpath_article, 'rb') as f:
                article = pk.load(f)
                with open(fpath_corpus, 'wb') as g:
                    pk.dump(article, g)
                    cnt += 1
        except Exception as e:
            errors.append((fname, e))

    print(f'  | Corpus: {cnt:,}')
    print(f'  | Errors: {len(errors):,}')


if __name__ == '__main__':
    ## Build corpus
    print('============================================================')
    print('Build corpus')

    errors = build_corpus()

    ## Save errors
    if errors:
        print('============================================================')
        print('Save errors')

        fpath_errors = os.path.sep.join((newspath.fdir_data, 'corpus_errors.json'))
        with open(fpath_errors, 'wb') as f:
            json.dump(errors, f)

        print(f'  | fdir : {newspath.fdir_data}')
        print('  | fname: corpus_errors.json')
    else:
        pass