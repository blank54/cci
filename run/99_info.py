#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
rootpath = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1])
sys.path.append(rootpath)

from newsutil import NewsPath, NewsIO, NewsFunc
newspath = NewsPath()
newsio = NewsIO()
newsfunc = NewsFunc()

import pickle as pk
from collections import defaultdict


# class NewsStatus:
#     '''
#     A class to print status of web crawling.

#     Methods
#     -------
#     queries
#         | Print the history of queries.
#         | The query files should start with "query_".
#     urls
#         | Print the number of url list.
#     articles
#         | Print the number of article list.
#     '''

#     def queries(self, fdir_queries):
#         history = defaultdict(list)

#         for fname in sorted(os.listdir(fdir_queries)):
#             if not fname.startswith('query_'):
#                 continue

#             collected_date = fname.replace('.txt', '').split('_')[1]
#             fpath_query = os.path.join(fdir_queries, fname)
#             query_list, date_list = NewsQueryParser().parse(fpath_query)

#             history['collected_date'].append(collected_date)
#             history['date_start'].append(date_list[0])
#             history['date_end'].append(date_list[-1])
#             history['num_query'].append(len(query_list))
#             history['query'].append(', '.join(query_list))

#         print('============================================================')
#         print('Status: Queries')
#         print('  | fdir: {}'.format(fdir_queries))
#         print('  | {:>13} {:>10} {:>10} {:>12} {:>15}'.format('CollectedDate', 'DateStart', 'DateEnd', 'NumOfQuery', 'Query'))
#         history_df = pd.DataFrame(history)
#         for i in range(len(history_df)):
#             collected_date = history_df.iloc[i]['collected_date']
#             date_start = history_df.iloc[i]['date_start']
#             date_end = history_df.iloc[i]['date_end']
#             num_query = history_df.iloc[i]['num_query']
#             query = history_df.iloc[i]['query']
#             print('  | {:>13} {:>10} {:>10} {:>12} {:>12}, ...'.format(collected_date, date_start, date_end, num_query, query[:10]))

#     def urls(self, fdir_urls):
#         urls = []
#         for fname in os.listdir(fdir_urls):
#             fpath_urls = os.path.join(fdir_urls, fname)
#             with open(fpath_urls, 'rb') as f:
#                 urls.extend(pk.load(f))

#         urls_distinct = list(set(urls))
#         print('============================================================')
#         print('Status: URLs')
#         print('  | fdir: {}'.format(fdir_urls))
#         print('  | Total # of urls: {:,}'.format(len(urls_distinct)))

#     def articles(self, fdir_articles):
#         flist = os.listdir(fdir_articles)

#         print('============================================================')
#         print('Status: Articles')
#         print('  | fdir: {}'.format(fdir_articles))
#         print('  | Total: {:,}'.format(len(flist)))


if __name__ == '__main__':
    ## URLs
    print('============================================================')
    print('URLs')

    flist_url = os.listdir(newspath.fdir_url_list)
    query2date = defaultdict(list)
    for fname_url in flist_url:
        query, date = newsfunc.parse_fname_url_list(fname_url)
        query2date[query].append(date)

    print('  | fdir: {}'.format(newspath.fdir_url_list))
    for query in query2date.keys():
        url_date_list = list(sorted([date for date in query2date[query]], reverse=False))

        print('--------------------------------------------------')
        print('    | query     : {}'.format(query))
        print('    | start date: {}'.format(url_date_list[0]))
        print('    | end date  : {}'.format(url_date_list[-1]))
        print('    | # of dates: {:,}'.format(len(query2date[query])))

    ## Articles
    print('============================================================')
    print('Articles')

    article_count = 0
    for fname_url in flist_url:
        fpath_url = os.path.join(newspath.fdir_url_list, fname_url)
        with open(fpath_url, 'rb') as f:
            article_count += len(pk.load(f))

    flist_article = []
    fsize_total_article = 0
    for path, dirs, files in os.walk(newspath.fdir_article):
        for f in files:
            fpath = os.path.sep.join((path, f))
            flist_article.append(fpath)
            fsize_total_article += os.path.getsize(fpath)

    print('  | fdir: {}'.format(newspath.fdir_article))
    print('  | # of articles(ready): {:,}'.format(article_count))
    print('  | # of articles(done) : {:,}'.format(len(flist_article)))
    print('  | total filesize      : {:,.02f} MB ({:,.02f} GB)'.format(fsize_total_article/(1024**2), fsize_total_article/(1024**3)))

    ## Corpus
    print('============================================================')
    print('Corpus')

    for fdir in sorted(os.listdir(os.path.sep.join((newspath.fdir_corpus, 'Q-건설')))):
        print('  | {}: {:>7,d} articles'.format(fdir, len(os.listdir(os.path.sep.join((newspath.fdir_corpus, 'Q-건설', fdir))))))