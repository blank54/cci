#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
import json
import psutil
import pickle as pk
from tqdm import tqdm


class NewsPath:
    root = os.path.dirname(os.path.abspath(__file__))

    fdir_query = os.path.sep.join((root, 'query'))
    fdir_data = os.path.sep.join((root, 'data'))
    fdir_corpus = os.path.sep.join((root, 'corpus'))
    fdir_model = os.path.sep.join((root, 'model'))

    fdir_url_list = os.path.sep.join((fdir_data, 'url_list'))
    fdir_article = os.path.sep.join((fdir_data, 'article'))


class NewsIO(NewsPath):
    def memory_usage(self):
        print('------------------------------------------------------------')
        active_memory = psutil.virtual_memory()._asdict()['used']
        print('  | Current memory usage: {:,.03f} GB ({:,.03f} MB)'.format(active_memory/(2**30), active_memory/(2**20)))
        print('--------------------------------------------------')

    def save_corpus(self, corpus, fname_corpus, verbose=True):
        fpath_corpus = os.path.sep.join((self.fdir_corpus, fname_corpus))
        with open(fpath_corpus, 'wb') as f:
            pk.dump(corpus, f)

        if verbose:
            print(f'  | fdir : {self.fdir_corpus}')
            print(f'  | fname: {fname_corpus}')

    def load_corpus(self, fname_corpus, verbose=True):
        fpath_corpus = os.path.sep.join((self.fdir_corpus, fname_corpus))
        with open(fpath_corpus, 'rb') as f:
            corpus = pk.load(f)

        if verbose:
            print(f'  | fdir : {self.fdir_corpus}')
            print(f'  | fname: {fname_corpus}')

        return corpus


class NewsFunc:
    def text2sents(self, text):
        '''
        text : a str object of doc.content
        '''

        sents = ['{}다.'.format(sent) for sent in text.split('다.')]
        return sents

    def parse_fname_url_list(self, fname_url_list):
        query_part, date_part = fname_url_list.replace('.pk', '').split('_')
        query = query_part.split('-')[-1]
        date = date_part.split('-')[-1]
        return query, date