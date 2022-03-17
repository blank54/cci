#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
rootpath = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1])
sys.path.append(rootpath)

from object import *
from newsutil import NewsPath, NewsFunc
query_parser = NewsQueryParser()
list_scraper = NaverNewsListScraper()
article_parser = NaverNewsArticleParser()
news_status = NewsStatus()
newspath = NewsPath()
newsfunc = NewsFunc()

import itertools
import pickle as pk
from tqdm import tqdm


## Parse query
def parse_query(fname_query):
    fpath_query = os.path.join(newspath.fdir_query, fname_query)
    query_list, date_list = query_parser.parse(fpath_query=fpath_query)
    return query_list, date_list

## Scrape URL list
def save_url_list(url_list, fname_url_list):
    fpath_url_list = os.path.join(newspath.fdir_url_list, fname_url_list)
    with open(fpath_url_list, 'wb') as f:
        pk.dump(url_list, f)

def scrape_url_list():
    global query_list, date_list

    print('============================================================')
    print('URL list scraping')
    for date in sorted(date_list, reverse=False):
        print('  | Date : {}'.format(date), end='')
        for query in query_list:
            print(' // Query: {}'.format(query))

            fname_url_list = 'Q-{}_D-{}.pk'.format(query, date)
            if fname_url_list in os.listdir(newspath.fdir_url_list):
                pass
            else:
                url_list = list_scraper.get_url_list(query=query, date=date)
                save_url_list(url_list, fname_url_list)

            print('  | >>> fdir : {}'.format(newspath.fdir_url_list))
            print('  | >>> fname: {}'.format(fname_url_list))
            print('  |')

## Parse article
def load_url_list(fname_url_list):
    fpath_url_list = os.path.join(newspath.fdir_url_list, fname_url_list)
    with open(fpath_url_list, 'rb') as f:
        url_list = pk.load(f)
    return url_list

def save_article(article, fpath_article):
    with open(fpath_article, 'wb') as f:
        pk.dump(article, f)

def load_article(fpath_article):
    with open(fpath_article, 'rb') as f:
        article = pk.load(f)
    return article

def parse_article(verbose_error):
    global query_list, date_list

    flist_url_list = []
    url_count = 0
    url_errors = []
    eof_errors = []
    for fname_url_list in os.listdir(newspath.fdir_url_list):
        query, date = newsfunc.parse_fname_url_list(fname_url_list=fname_url_list)
        if query not in query_list:
            continue
        elif date not in date_list:
            continue
        else:
            flist_url_list.append(fname_url_list)
            url_count += len(load_url_list(fname_url_list=fname_url_list))

    print('============================================================')
    print('Article parsing')
    with tqdm(total=url_count) as pbar:
        for fname_url_list in flist_url_list:
            query_list_of_url, _ = query_parser.urlname2query(fname_url_list=fname_url_list)
            url_list = load_url_list(fname_url_list=fname_url_list)

            for url in url_list:
                pbar.update(1)

                fname_article = 'a-{}.pk'.format(article_parser.url2id(url))
                fpath_article = os.path.join(newspath.fdir_article, fname_article)
                if not os.path.isfile(fpath_article):
                    try:
                        article = article_parser.parse(url=url)
                    except:
                        url_errors.append(url)
                        continue
                else:
                    try:
                        article = load_article(fpath_article=fpath_article)
                    except EOFError as e:
                        eof_errors.append(e)
                        continue

                article.extend_query(query_list_of_url)
                save_article(article=article, fpath_article=fpath_article)

    print('============================================================')
    print('  | Initial      : {:,} urls'.format(url_count))
    print('  | Done         : {:,} articles'.format(len(os.listdir(newspath.fdir_article))))
    print('  | Error in URL : {:,}'.format(len(url_errors)))
    print('  | Error in File: {:,}'.format(len(eof_errors)))

    if verbose_error and any((url_errors, eof_errors)):
        print('------------------------------------------------------------')
        print('Errors on URLs:')
        for url in url_errors:
            print(url)

        print('------------------------------------------------------------')
        print('Errors in EOFs:')
        for e in eof_errors:
            print(e)

## Use article
def print_article(fpath_article):
    article = load_article(fpath_article=fpath_article)
    print('============================================================')
    print('Article information')
    print('  | URL     : {}'.format(article.url))
    print('  | Title   : {}'.format(article.title))
    print('  | Date    : {}'.format(article.date))
    print('  | Query   : {}'.format(', '.join(article.query)))
    print('  | Category: {}'.format(article.category))
    print('  | Content : {}...'.format(article.content[:20]))


if __name__ == '__main__':
    query_id = str(sys.argv[1])
    option_scrape_url_list = str(sys.argv[2])
    option_parse_article = str(sys.argv[3])

    ## Web crawling information
    fname_query = 'query_{}.txt'.format(query_id)
        
    ## Parse query
    query_list, date_list = parse_query(fname_query=fname_query)

    ## Scrape URL list
    if option_scrape_url_list == 'true':
        scrape_url_list()
    else:
        pass

    ## Parse article
    if option_parse_article == 'true':
        parse_article(verbose_error=True)
    else:
        pass

    # ## Data collection status
    # news_status.queries(fdir_query=fdir_query)
    # news_status.urls(fdir_urls=newspath.fdir_url_list)
    # news_status.articles(newspath.fdir_articles=newspath.fdir_article)

    # ## Use article
    # fpath_article = os.path.join(newspath.fdir_article, os.listdir(newspath.fdir_article)[0])
    # print_article(fpath_article=fpath_article)