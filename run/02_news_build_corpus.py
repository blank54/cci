#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
rootpath = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1])
sys.path.append(rootpath)

from news import NewsPath, NewsIO
newspath = NewsPath()

import re
import json
from glob import glob
from tqdm import tqdm


def build_corpus(**kwargs):
    cnt = 0
    errors = []
    for fname_article in tqdm(os.listdir(newspath.fdir_articles)):
        fpath_article = os.path.sep.join((newspath.fdir_articles, fname_article))
        with open(fpath_article, 'r', encoding='utf-8') as f:
            article = json.load(f)
            article['yearmonth'] = ''.join(re.split('\.', article['date'])[:2])
            fpath_article_corpus = os.path.sep.join((newspath.fdir_corpus, article['yearmonth'], fname_article))
            article['fpath_article_corpus'] = fpath_article_corpus
            try:
                os.makedirs(os.path.dirname(fpath_article_corpus), exist_ok=False)
            except FileExistsError:
                pass

            with open(fpath_article_corpus, 'w', encoding='utf-8') as g:
                json.dump(article, g)
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

        fname_errors = 'errors_build_corpus.json'
        fpath_errors = os.path.sep.join((newspath.fdir_data, fname_errors))
        with open(fpath_errors, 'w', encoding='utf-8') as f:
            json.dump(errors, f)

        print(f'  | fdir : {newspath.fdir_data}')
        print(f'  | fname: {fname_errors}')
    else:
        pass