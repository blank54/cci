#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
import json
import psutil
import pickle as pk
from tqdm import tqdm
from datetime import datetime


class NewsPath:
    root = os.path.dirname(os.path.abspath(__file__))

    fdir_query = os.path.sep.join((root, 'query'))
    fdir_data = os.path.sep.join((root, 'data'))
    fdir_articles = os.path.sep.join((root, 'articles'))
    fdir_corpus = os.path.sep.join((root, 'corpus'))
    fdir_corpus_monthly = os.path.sep.join((root, 'corpus_monthly'))
    fdir_model = os.path.sep.join((root, 'model'))
    fdir_thesaurus = os.path.sep.join((root, 'thesaurus'))

    fdir_url_list = os.path.sep.join((fdir_data, 'url_list'))
    fdir_article = os.path.sep.join((fdir_data, 'article'))


class NewsIO(NewsPath):
    def memory_usage(self):
        print('------------------------------------------------------------')
        active_memory = psutil.virtual_memory()._asdict()['used']
        print('  | Current memory usage: {:,.03f} GB ({:,.03f} MB)'.format(active_memory/(2**30), active_memory/(2**20)))
        print('--------------------------------------------------')

    def save(self, _object, _type, fname_object, verbose=True):
        fdir_object = os.path.sep.join((self.root, _type))
        fpath_object = os.path.sep.join((fdir_object, fname_object))

        with open(fpath_object, 'wb') as f:
            pk.dump(_object, f)

        if verbose:
            print(f'  | fdir : {fdir_object}')
            print(f'  | fname: {fname_object}')

    def load(self, fname_object, _type, verbose=True):
        fdir_object = os.path.sep.join((self.root, _type))
        fpath_object = os.path.sep.join((fdir_object, fname_object))
        with open(fpath_object, 'rb') as f:
            _object = pk.load(f)

        if verbose:
            print(f'  | fdir : {fdir_object}')
            print(f'  | fname: {fname_object}')

        return _object

    def save_corpus_monthly(self, article):
        yearmonth = datetime.strptime(article.date.datetime, '%Y.%m.%d').strftime('%Y%m')
        fpath_corpus_monthly = os.path.sep.join((self.fdir_corpus_monthly, yearmonth, article.fname))

        try:
            os.makedirs(os.path.dirname(fpath_corpus_monthly), exist_ok=False)
        except FileExistsError:
            pass

        with open(fpath_corpus_monthly, 'wb') as g:
            pk.dump(article, g)

    def read_thesaurus(self, fname_thesaurus):
        fpath_thesaurus = os.path.sep.join((self.fdir_thesaurus, fname_thesaurus))
        with open(fpath_thesaurus, 'r', encoding='utf-8') as f:
            word_list = list(set([w.strip() for w in f.read().strip().split('\n')]))

        return word_list


class NewsFunc(NewsPath):
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