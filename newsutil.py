#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import psutil
import pickle as pk
from tqdm import tqdm


class NewsPath:
    root = os.path.dirname(os.path.abspath(__file__))

    fdir_query = os.path.sep.join((root, 'query'))
    fdir_data = os.path.sep.join((root, 'data'))
    fdir_corpus = os.path.sep.join((root, 'corpus'))

    fdir_url_list = os.path.sep.join((fdir_data, 'url_list'))
    fdir_article = os.path.sep.join((fdir_data, 'article'))


class NewsIO(NewsPath):
    def memory_usage(self):
        print('------------------------------------------------------------')
        active_memory = psutil.virtual_memory()._asdict()['used']
        print('  | Current memory usage: {:,.03f} GB ({:,.03f} MB)'.format(active_memory/(2**30), active_memory/(2**20)))
        print('--------------------------------------------------')

    def read_articles(self):
        print('============================================================')
        print('Read articles')

        flist = os.listdir(self.fdir_article)
        docs = []
        with tqdm(total=len(flist)) as pbar:
            for fname in flist:
                fpath = os.path.join(self.fdir_article, fname)
                with open(fpath, 'rb') as f:
                    docs.append(pk.load(f))
                    pbar.update(1)

        print('  | fdir: {}'.format(self.fdir_article))
        print('  | # of articles: {:,}'.format(len(flist)))
        return docs

    def save_corpus(self, corpus, fname):
        print('============================================================')
        print('Save corpus')

        fpath = os.path.join(self.fdir_corpus, fname)
        with open(fpath, 'wb') as f:
            pk.dump(corpus, f)

        print('  | fdir : {}'.format(self.fdir_corpus))
        print('  | fname: {}'.format(fname))


class NewsFunc:
    def text2sents(self, text):
        '''
        text : a str object of doc.content
        '''

        sents = ['{}다.'.format(sent) for sent in text.split('다.')]
        return sents
