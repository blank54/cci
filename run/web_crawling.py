#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
rootpath = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1])
sys.path.append(rootpath)

from object import NewsQueryParser, NaverNewsListScraper, NaverNewsArticleParser, NewsStatus
from newsutil import NewsPath
query_parser = NewsQueryParser()
list_scraper = NaverNewsListScraper()
article_parser = NaverNewsArticleParser()
news_status = NewsStatus()
newspath = NewsPath()

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

def scrape_url_list(query_list, date_list):
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

def parse_article(verbose_error=False):
    total_num_urls = len(list(itertools.chain(*[load_url_list(fname) for fname in os.listdir(newspath.fdir_url_list)])))
    errors = []

    print('============================================================')
    print('Article parsing')
    with tqdm(total=total_num_urls) as pbar:
        for fname_url_list in os.listdir(newspath.fdir_url_list):
            query_list, _ = query_parser.urlname2query(fname_url_list=fname_url_list)
            url_list = load_url_list(fname_url_list=fname_url_list)

            for url in url_list:
                pbar.update(1)

                fname_article = 'a-{}.pk'.format(article_parser.url2id(url))
                fpath_article = os.path.join(newspath.fdir_article, fname_article)
                if not os.path.isfile(fpath_article):
                    try:
                        article = article_parser.parse(url=url)
                    except:
                        errors.append(url)
                        continue
                else:
                    article = load_article(fpath_article=fpath_article)

                article.extend_query(query_list)
                save_article(article=article, fpath_article=fpath_article)

    print('============================================================')
    print('  | Initial   : {:,} urls'.format(total_num_urls))
    print('  | Done      : {:,} articles'.format(len(os.listdir(newspath.fdir_article))))
    print('  | Error     : {:,}'.format(len(errors)))

    if verbose_error and errors:
        print('============================================================')
        print('Errors on articles:')
        for url in errors:
            print(url)

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
    ## Web crawling information
    fname_query = 'query_20211006-2.txt'
        
    ## Parse query
    query_list, date_list = parse_query(fname_query=fname_query)

    ## Scrape URL list
    scrape_url_list(query_list=query_list, date_list=date_list)

    ## Parse article
    parse_article()

    # ## Data collection status
    # news_status.queries(fdir_query=fdir_query)
    # news_status.urls(fdir_urls=newspath.fdir_url_list)
    # news_status.articles(newspath.fdir_articles=newspath.fdir_article)

    # ## Use article
    # fpath_article = os.path.join(newspath.fdir_article, os.listdir(newspath.fdir_article)[0])
    # print_article(fpath_article=fpath_article)