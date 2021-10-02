#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os


class MarketPath:
    root = os.path.dirname(os.path.abspath(__file__))

    fdir_query = os.path.sep.join((root, 'query'))
    fdir_data = os.path.sep.join((root, 'data'))

    fdir_url_list = os.path.sep.join((fdir_data, 'url_list'))
    fdir_article = os.path.sep.join((fdir_data, 'article'))