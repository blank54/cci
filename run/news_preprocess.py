#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
rootpath = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1])
sys.path.append(rootpath)

from newsutil import NewsIO
newsio = NewsIO()


for idx, article in newsio.read_articles(iter='each'):
    if idx == 5:
        break
    else:
        print(article.url)
        print(article.id)
        print(article.query)
        print(article.title)
        print(article.date)
        print(article.category)
        print(article.content)