#!/usr/bin/env python -W ignore::DeprecationWarning
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
sys.path.append(os.getcwd())

from news import NewsDate

import time
import itertools
import numpy as np
from datetime import datetime, timedelta

from urllib import request
from urllib.parse import quote
from bs4 import BeautifulSoup


class NewsQueryParser:
    '''
    A class of news query parser.

    Method
    ------
    return_query_list
        | Parse the query file and return the list of queries.
    return_date_list
        | Parse the query file and return the list of dates.
    parse
        | Parse the query file and return the list of queries and dates.
    urlname2query
        | Extract query from url name.
    '''

    def return_query_list(self, query_file):
        '''
        A method to parse the query file and return the list of queries.

        Attributes
        ----------
        query_file : str
            | The query file read by "with open".
        '''

        _splitted_queries = [queries.split('\n') for queries in query_file.split('\n\n')[1:]]
        _queries_combs = list(itertools.product(*_splitted_queries))
        query_list = ['+'.join(e) for e in _queries_combs]
        return query_list

    def return_date_list(self, query_file):
        '''
        A method to parse the query file and return the list of dates.

        Attributes
        ----------
        query_file : str
            | The query file read by "with open".
        '''

        date_start, date_end = query_file.split('\n\n')[0].split('\n')

        date_start_formatted = datetime.strptime(date_start, '%Y%m%d')
        date_end_formatted = datetime.strptime(date_end, '%Y%m%d')
        delta = date_end_formatted - date_start_formatted

        date_list = []
        for i in range(delta.days+1):
            day = date_start_formatted + timedelta(days=i)
            date_list.append(datetime.strftime(day, '%Y%m%d'))
        return date_list

    def parse(self, fpath_query):
        '''
        A method to parse the query file and return the list of queries and dates.

        Attributes
        ----------
        fpath_query : str
            | The filepath of the query.
        '''

        with open(fpath_query, 'r', encoding='utf-8') as f:
            query_file = f.read()

        query_list = self.return_query_list(query_file=query_file)
        date_list = self.return_date_list(query_file=query_file)
        return query_list, date_list

    def urlname2query(self, fname_url_list):
        '''
        A method to extract query from given url name.

        Attributes
        fname_url_list : str
            | The file name of the url list.
        '''

        Q, D = fname_url_list.replace('.pk', '').split('_')
        query_list = Q.split('-')[1].split('+')
        date = D.split('-')[1]
        return query_list, date


class NewsQuery:
    '''
    A class of news query to address the encoding issues.

    Attributes
    ----------
    query : str
        | Query in string format.
    '''

    def __init__(self, query):
        self.query = query

    def __call__(self):
        return quote(self.query.encode('utf-8'))

    def __str__(self):
        return '{}'.format(self.query)

    def __len__(self):
        return len(self.query.split('+'))


class NewsCrawler():
    '''
    A class of news crawler that includes headers.

    Attributes
    ----------
    time_lag_random : float
        | A random number for time lag.
    headers : dict
        | Crawling header that is used generally.
    '''

    time_lag_random = np.random.normal(loc=1.0, scale=0.1)
    headers = {'User-Agent': '''
        [Windows64,Win64][Chrome,58.0.3029.110][KOS] 
        Mozilla/5.0 Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) 
        Chrome/58.0.3029.110 Safari/537.36
        '''}


class NaverNewsListScraper(NewsCrawler):
    '''
    A scraper for the article list of naver news.

    Attributes
    ----------
    url_base : str
        | The url base of the article list page of naver news.

    Methods
    -------
    get_url_list
        | Get the url list of articles for the given query and dates.
    '''

    def __init__(self):
        self.url_base = 'https://search.naver.com/search.naver?where=news&sm=tab_pge&query={}&sort=1&photo=0&field=0&pd=3&ds={}&de={}&mynews=0&office_type=0&office_section_code=0&news_office_checked=&nso=so:dd,p:from{}to{},a:all&start={}'

    def get_url_list(self, query, date):
        '''
        A method to get url list of articles for the given query and dates.

        Attributes
        ----------
        query : str
            | A query of simple string format.
        date : str
            | A date to search the query. (foramt : yyyymmdd)
        '''

        query = NewsQuery(query)
        date = NewsDate(date)

        url_list = []
        start_idx = 1
        while True:
            url_list_page = self.url_base.format(query(), date(), date(), date.date, date.date, start_idx)
            req = request.Request(url=url_list_page, headers=self.headers)
            html = request.urlopen(req).read()
            soup = BeautifulSoup(html, 'lxml')
            time.sleep(self.time_lag_random)

            url_list.extend([s.get('href') for s in soup.find_all('a', class_='info') if '네이버뉴스' in s])
            start_idx += 10

            print('\r  | >>> Parsing {:,} urls'.format(len(url_list)), end='')

            if soup.find('div', class_='not_found02'):
                break
            else:
                continue

        print()
        return list(set(url_list))


class NaverNewsArticleParser(NewsCrawler):
    '''
    A parser of naver news article page.

    Methods
    -------
    parse
        | Parse the page of the given url and return the article information.
    url2id
        | Extract article id from the url.
    '''

    def __init__(self):
        pass

    def parse(self, url):
        '''
        A method to parse the article page.

        Attributes
        ----------
        url : str
            | The url of the article page.
        '''

        req = request.Request(url=url, headers=self.headers)
        html = request.urlopen(req).read()
        soup = BeautifulSoup(html, 'lxml')
        time.sleep(self.time_lag_random)

        title = soup.find_all('h3', {'id': 'articleTitle'})[0].get_text().strip()
        date = soup.find_all('span', {'class': 't11'})[0].get_text().split()[0].replace('.', '').strip()
        content = soup.find_all('div', {'id': 'articleBodyContents'})[0].get_text().strip()

        try:
            category = soup.find_all('em', {'class': 'guide_categorization_item'})[0].get_text().strip()
        except IndexError:
            category = None

        article = {'url': url,
                   'id': self.url2id(url),
                   'title': title,
                   'date': date,
                   'category': category,
                   'content': content}
        return article

    def url2id(self, url):
        '''
        A method to extract article id from the url.

        Attributes
        ----------
        url : str
            | The article url.
        '''

        id = str(url.split('=')[-1])
        return id