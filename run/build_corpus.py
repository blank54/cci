#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
rootpath = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1])
sys.path.append(rootpath)

from newsutil import NewsIO
newsio = NewsIO()

import psutil
import numpy as np


if __name__ == '__main__':
    newsio.memory_usage()

    ## Read articles as docs
    docs = newsio.read_articles()
    newsio.memory_usage()

    ## Save docs as corpus
    fname_corpus = 'corpus_20211007.pk'
    newsio.save_corpus(corpus=docs, fname=fname_corpus)
    newsio.memory_usage()