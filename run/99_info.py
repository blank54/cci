#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
rootpath = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1])
sys.path.append(rootpath)

from news import NewsCorpus, NumericMeta, NumericData, NewsPath, NewsIO, NewsFunc
newspath = NewsPath()
newsio = NewsIO()
newsfunc = NewsFunc()

import pickle as pk
from collections import defaultdict


if __name__ == '__main__':
    # ## URLs
    # print('============================================================')
    # print('URLs')

    # flist_url = os.listdir(newspath.fdir_url_list)
    # query2date = defaultdict(list)
    # for fname_url in flist_url:
    #     query, date = newsfunc.parse_fname_url_list(fname_url)
    #     query2date[query].append(date)

    # print('--------------------------------------------------')
    # print('  | fdir: {}'.format(newspath.fdir_url_list))
    # for query in query2date.keys():
    #     url_date_list = list(sorted([date for date in query2date[query]], reverse=False))

    #     print('    | query     : {}'.format(query))
    #     print('    | start date: {}'.format(url_date_list[0]))
    #     print('    | end date  : {}'.format(url_date_list[-1]))
    #     print('    | # of dates: {:,}'.format(len(query2date[query])))

    # ## Articles
    # print('============================================================')
    # print('Articles')

    # article_count = 0
    # for fname_url in flist_url:
    #     fpath_url = os.path.join(newspath.fdir_url_list, fname_url)
    #     with open(fpath_url, 'rb') as f:
    #         article_count += len(pk.load(f))

    # flist_article = []
    # fsize_total_article = 0
    # for path, dirs, files in os.walk(newspath.fdir_article):
    #     for f in files:
    #         fpath = os.path.sep.join((path, f))
    #         flist_article.append(fpath)
    #         fsize_total_article += os.path.getsize(fpath)

    # print('  | fdir: {}'.format(newspath.fdir_article))
    # print('  | # of articles in url list: {:,}'.format(article_count))
    # print('  | # of articles collected  : {:,}'.format(len(flist_article)))
    # print('  | total filesize           : {:,.02f} MB ({:,.02f} GB)'.format(fsize_total_article/(1024**2), fsize_total_article/(1024**3)))

    # ## Corpus
    # print('============================================================')
    # print('Corpus')

    # corpus = NewsCorpus(dname_corpus='corpus', start='200501', end='201912')
    # corpus_filtered = NewsCorpus(dname_corpus='corpus_topic_filtered', start='200501', end='201912')
    
    # print('  | Original corpus: {:,}'.format(len(corpus)))
    # print('  | Filtered corpus: {:,}'.format(len(corpus_filtered)))

    ## Numeric Data
    print('============================================================')
    print('Numeric data')

    numeric_meta = NumericMeta(fname='metadata_numeric.txt', fdir=newspath.fdir_thesaurus)

    fdir_numeric_data = os.path.sep.join((newspath.fdir_data, 'numeric'))
    numeric_data = NumericData(fdir=fdir_numeric_data)
    numeric_df = numeric_data.to_df(start='200501', end='201912')

    print('--------------------------------------------------')    
    for kor, eng, abb in numeric_meta.info:
        num_vars = len([var for var in numeric_data.vars if abb.lower() == var.split('_')[0]])
        print('  | {}: {} ({}) -> ({:,})'.format(kor, eng, abb, num_vars))

    print('--------------------------------------------------')
    print('  | Num of indicators       : {:,}'.format(numeric_data.num_indicators))
    print('  | Num of variables (total): {:,}'.format(numeric_data.num_vars))
    print('  | Shape of dataset        : {}'.format(numeric_df.shape))

    print('--------------------------------------------------')
    print(numeric_df.head())