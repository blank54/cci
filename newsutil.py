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

    fdir_url_list = os.path.sep.join((fdir_data, 'url_list'))
    fdir_article = os.path.sep.join((fdir_data, 'article'))


class NewsIO(NewsPath):
    def memory_usage(self):
        print('------------------------------------------------------------')
        active_memory = psutil.virtual_memory()._asdict()['used']
        print('  | Current memory usage: {:,.03f} GB ({:,.03f} MB)'.format(active_memory/(2**30), active_memory/(2**20)))
        print('--------------------------------------------------')

    def read_articles(self, iter='all'):
        flist = os.listdir(self.fdir_article)

        if iter == 'all':
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

        elif iter == 'each':
            for idx, fname in enumerate(flist):
                fpath = os.path.join(self.fdir_article, fname)
                with open(fpath, 'rb') as f:
                    yield (idx, pk.load(f))

        else:
            print('ArgvError: Use proper value of \"iter\"')
            sys.exit()

    def save_corpus(self, corpus, fpath):
        with open(fpath, 'wb') as f:
            json.dump(corpus, f)


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