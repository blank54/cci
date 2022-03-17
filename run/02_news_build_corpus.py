#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
rootpath = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1])
sys.path.append(rootpath)

from newsutil import NewsPath
newspath = NewsPath()

import random
import pickle as pk
from tqdm import tqdm


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

def save_corpus(corpus, fname_corpus):
    fpath_corpus = os.path.sep.join((newspath.fdir_corpus, fname_corpus))
    with open(fpath_corpus, 'wb') as f:
        pk.dump(corpus, f)

    print(f'  | fdir : {newspath.fdir_corpus}')
    print(f'  | fname: {fname_corpus}')


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

    save_corpus(corpus=corpus, fname_corpus=fname_corpus)