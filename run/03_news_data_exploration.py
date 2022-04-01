#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
rootpath = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1])
sys.path.append(rootpath)

from object import NewsCorpus, NewsDate
from newsutil import NewsPath
newspath = NewsPath()

from copy import deepcopy
from collections import defaultdict
import matplotlib.pyplot as plt


def assign_articles_monthly(corpus):
    data = defaultdict(list)
    for doc in corpus.iter():
        yearmonth = NewsDate(date=doc.date).yearmonth
        data[yearmonth].append(doc)

    data = deepcopy(list(sorted(data.items(), key=lambda x:x[0], reverse=False)))

    return data

def visualize_news(data):
    yearmonth_list, count_list = [], []
    for yearmonth, docs in data:
        yearmonth_list.append(yearmonth)
        count_list.append(len(docs))

    plt.plot(yearmonth_list, count_list)
    plt.show()


if __name__ == '__main__':
    ## Data Import
    print('============================================================')
    print('Load corpus')

    corpus = NewsCorpus(fdir_corpus=newspath.fdir_corpus)

    ## Article Frequency
    print('============================================================')
    print('Assign articles')

    data = assign_articles_monthly(corpus)

    print(f'  | yearmonths: {len(data):,}')

    ## Visualization
    print('============================================================')
    print('Visualize article occurrences')

    visualize_news(data=data)