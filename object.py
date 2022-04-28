#!/usr/bin/env python -W ignore::DeprecationWarning
# -*- coding: utf-8 -*-

# Configuration
import os
import random
import pickle as pk
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime

import gensim.corpora as corpora
from gensim.models.ldamodel import LdaModel
from gensim.models import CoherenceModel
import pyLDAvis.gensim


class NewsArticle:
    '''
    A class of news article.

    Attributes
    ----------
    url : str
        | The article url.
    id : str
        | The identification code for the article.
    query : list
        | A list of queries that were used to search the article.
    title : str
        | The title of the article.
    date : str
        | The uploaded date of the article. (format : yyyymmdd)
    category : str
        | The category that the article belongs to.
    content : str
        | The article content.
    content_normalized : str
        | Normalized content of the article.

    Methods
    -------
    extend_query
        | Extend the query list with the additional queries that were used to search the article.
    '''

    def __init__(self, **kwargs):
        self.url = kwargs.get('url', '')
        self.id = kwargs.get('id', '')
        self.query = []

        self.title = kwargs.get('title', '')
        self.date = kwargs.get('date', '')
        self.category = kwargs.get('category', '')
        self.content = kwargs.get('content', '')

        self.preprocess = False
        self.sents = kwargs.get('sents', '')
        self.normalized_sents = kwargs.get('normalized_sents', '')
        self.nouns = kwargs.get('nouns', '')
        self.nouns_stop = kwargs.get('nouns_stop', '')

    def extend_query(self, query_list):
        '''
        A method to extend the query list.

        Attributes
        ----------
        query_list : list
            | A list of queries to be extended.
        '''

        queries = self.query
        queries.extend(query_list)
        self.query = list(set(queries))


class NewsDate:
    '''
    A class of news dates to address the encoding issues.

    Attributes
    ----------
    date : str
        | Date in string format. (format : yyyymmdd)
    formatted : datetime
        | Formatted date.
    '''

    def __init__(self, date):
        self.date = date
        self.yearmonth = self.__yearmonth()
        self.datetime = self.__datetime()

    def __call__(self):
        return self.datetime

    def __str__(self):
        return '{}'.format(self.__call__())

    def __datetime(self):
        try:
            return datetime.strptime(self.date, '%Y%m%d').strftime('%Y.%m.%d')
        except:
            return ''

    def __yearmonth(self):
        return datetime.strptime(self.date, '%Y%m%d').strftime('%Y%m')
        

class NewsCorpus:
    def __init__(self, fdir_corpus, **kwargs):
        self.fdir_corpus = fdir_corpus

    def __len__(self):
        return len(os.listdir(self.fdir_corpus))

    def iter(self, **kwargs):
        flist = os.listdir(self.fdir_corpus)
        for fname in tqdm(flist):
            fpath = os.path.sep.join((self.fdir_corpus, fname))
            try:
                with open(fpath, 'rb') as f:
                    yield pk.load(f)
            except:
                print(f'UnpicklingError: {fname}')

    def iter_sampling(self, n, random_state=42):
        flist = random.sample(os.listdir(self.fdir_corpus), k=n)
        for fname in tqdm(flist):
            fpath = os.path.sep.join((self.fdir_corpus, fname))
            with open(fpath, 'rb') as f:
                yield pk.load(f)


class NewsMonthlyCorpus:
    def __init__(self, fdir_corpus, **kwargs):
        self.fdir_corpus = fdir_corpus
        self.flist = sorted(os.listdir(self.fdir_corpus), reverse=False)

        self.start = kwargs.get('start', self.flist[0])
        self.end = kwargs.get('end', self.flist[-1])
        self.yearmonth_list = self.__get_yearmonth_list()

    def __len__(self):
        return len(self.yearmonth_list)

    def __get_yearmonth_list(self):
        yearmonth_start = datetime.strptime(self.start, '%Y%m').strftime('%Y-%m-%d')
        yearmonth_end = datetime.strptime(self.end, '%Y%m').strftime('%Y-%m-%d')
        return pd.date_range(yearmonth_start, yearmonth_end, freq='MS').strftime('%Y%m').tolist()

    def iter_monthly(self):
        for yearmonth in tqdm(self.yearmonth_list):
            fdir_corpus_yearmonth = os.path.sep.join((self.fdir_corpus, yearmonth))
            corpus_yearmonth = []
            for fname in os.listdir(fdir_corpus_yearmonth):
                fpath = os.path.sep.join((fdir_corpus_yearmonth, fname))
                with open(fpath, 'rb') as f:
                    corpus_yearmonth.append(pk.load(f))

            yield corpus_yearmonth

    def iter(self):
        for yearmonth in tqdm(self.yearmonth_list):
            fdir_corpus_yearmonth = os.path.sep.join((self.fdir_corpus, yearmonth))
            for fname in os.listdir(fdir_corpus_yearmonth):
                fpath = os.path.sep.join((fdir_corpus_yearmonth, fname))
                with open(fpath, 'rb') as f:
                    yield pk.load(f)


class Word:
    def __init__(self, word):
        self.word = word

        self.tf = ''
        self.df = ''
        self.idf = ''
        self.tfidf = ''

    def __str__(self):
        return word


class NumericMeta:
    def __init__(self, fname, fdir):
        self.fname = fname
        self.fdir = fdir
        self.fpath = os.path.sep.join((self.fdir, self.fname))

        self.info = self.__read_metadata()
        self.kor2eng = {k: e for k, e, a in self.info}
        self.kor2abb = {k: a for k, e, a in self.info}
        self.eng2kor = {e: k for k, e, a in self.info}
        self.eng2abb = {e: a for k, e, a in self.info}
        self.abb2kor = {a: k for k, e, a in self.info}
        self.abb2eng = {a: e for k, e, a in self.info}

    def __read_metadata(self):
        with open(self.fpath, 'r', encoding='utf-8') as f:
            return [row.split('  ') for row in f.read().strip().split('\n')]

    def __len__(self):
        return len(self.info)

    def __str__(self):
        return self.info


class NumericData():
    def __init__(self, fdir, **kwargs):
        self.fdir = fdir

        self.data_list = self.__read_data()

        self.num_vars = len(os.listdir(self.fdir))
        self.num_attrs = len(self.data_list)

        self.start = kwargs.get('start', 'InputRequired')
        self.end = kwargs.get('end', 'InputRequired')

    def __read_data(self):
        data_list = []
        for fname in os.listdir(self.fdir):
            fpath = os.path.sep.join((self.fdir, fname))

            _df = pd.read_excel(fpath, na_values='')
            for _, row in _df.iterrows():
                year = row['yearmonth'].year
                month = row['yearmonth'].month
                yearmonth = f'{year}{month:02}'

                if yearmonth == 'nannan':
                    print(row)

                for attr in row.keys():
                    if attr == 'yearmonth':
                        continue
                    else:
                        data_list.append((yearmonth, attr, row[attr]))

        return data_list

    def __set_time_range(self, start, end):
        '''
        Attributes
        ----------
        start : str
            | YYYYMM
        end : str
            |YYYYMM
        '''

        start_dt = datetime.strptime(start, '%Y%m')
        end_dt = datetime.strptime(end, '%Y%m')
        return pd.date_range(start_dt, end_dt, freq='MS').strftime('%Y%m').tolist()

    def to_df(self, **kwargs):
        start = kwargs.get('start', self.start)
        end = kwargs.get('end', self.end)
        time_range = self.__set_time_range(start, end)

        _dict = defaultdict(list)
        _dict['yearmonth'] = time_range
        for yearmonth, attr, value in sorted(self.data_list, key=lambda x:x[0], reverse=False):
            if yearmonth in time_range:
                _dict[attr].append(value)
            else:
                continue

        _df = pd.DataFrame(_dict)
        return _df