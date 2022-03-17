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

import random
import pickle as pk
from tqdm import tqdm
random.seed(42)

def build_corpus(**kwargs):
    print('--------------------------------------------------')
    print('Find articles')

    flist = []
    for path, dirs, files in os.walk(newspath.fdir_article):
        flist.extend([os.path.sep.join((path, fname)) for fname in files])

    print(f'  | Total: {len(flist):,}')

    SAMPLE_SIZE = kwargs.get('SAMPLE_SIZE', '')
    if SAMPLE_SIZE:
        flist = random.sample(flist, SAMPLE_SIZE)
        print(f'  | Sample size: {SAMPLE_SIZE:,}')
    else:
        pass

    print('--------------------------------------------------')
    print('Articles -> Corpus')

    corpus = []
    for fpath in tqdm(flist):
        with open(fpath, 'rb') as f:
            corpus.append(pk.load(f))

    print(f'  | Corpus: {len(corpus):,}')

    return corpus


if __name__ == '__main__':
    ## Parameters
    SAMPLE_SIZE = 1000

    ## Filenames
    fname_corpus = f'corpus_{str(SAMPLE_SIZE)}.pk'

    ## Build corpus
    print('============================================================')
    print('Build corpus')

    corpus = build_corpus(SAMPLE_SIZE=SAMPLE_SIZE)

    ## Save corpus
    print('============================================================')
    print('Save corpus')

    newsio.save(_object=corpus, _type='corpus', fname_object=fname_corpus)