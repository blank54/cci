#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
rootpath = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1])
sys.path.append(rootpath)

from object import NewsCorpus
from newsutil import NewsIO, NewsPath
newsio = NewsIO()
newspath = NewsPath()

import re
import pickle as pk
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from collections import defaultdict

from konlpy.tag import Komoran
komoran = Komoran()


def normalize_text(text):
    text = deepcopy(re.sub('[^a-zA-Z0-9ㄱ-ㅣ가-힣\s\(\)\.]', '', text))
    text = deepcopy(re.sub('\s+', ' ', text).strip())
    text = deepcopy(re.sub('\.+', '.', text).strip())
    return text

def parse_sent(text):
    idx_list = [int(m.start()+1) for m in re.finditer('[ㄱ-힣]\.', text)]
    sents = deepcopy([c if i not in idx_list else '  SEP  ' for i, c in enumerate(text)])
    sents = deepcopy([s.strip() for s in ''.join(sents).split('  SEP  ')])
    return sents

def concatenate_short_sent(sents, MIN_SENT_LEN):
    idx_list = [i for i, s in enumerate(sents) if len(s.split()) < MIN_SENT_LEN]

    output_sents = []
    pre = []
    for idx, sent in enumerate(sents):
        if idx not in idx_list:
            if pre:
                pre.append(sent)
                sent = ' '.join(pre)
                pre = deepcopy([])
            else:
                pass
            output_sents.append(sent)
        else:
            pre.append(sent)
            continue
    
    return output_sents

def remove_stopwords(sent, stoplist):
    return [w for w in sent if w not in stoplist]

def preprocess(corpus):
    _start = datetime.now()
    for article in corpus.iter():
        ## Preprocess
        normalized_text = normalize_text(text=article.content)
        sents = parse_sent(text=normalized_text)
        concatenated_sents = concatenate_short_sent(sents, MIN_SENT_LEN=MIN_SENT_LEN)

        for sent in concatenated_sents:
            trash_score = sum([1 if word in sent else 0 for word in trash_word_list])
            if trash_score < MAX_TRASH_SCORE:
                morphs = komoran.nouns(sent)

                article.normalized_sents.append(sent)
                article.nouns.append(morphs)
                article.nouns_stop.append(remove_stopwords(sent=morphs, stoplist=stoplist))
            else:
                continue

        ## Save corpus
        article.preprocess = True
        fpath_article_preprocessed = os.path.sep.join((newspath.fdir_corpus, article.fname))
        with open(fpath_article_preprocessed, 'wb') as f:
            pk.dump(article, f)

        ## Save corpus monthly
        newsio.save_corpus_monthly(article=article)


if __name__ == '__main__':
    ## Filenames
    fname_trash_words = 'trashlist_20220614.txt'
    fname_stoplist = 'stoplist_20220614.txt'

    ## Parameters
    MIN_SENT_LEN = 3
    MAX_TRASH_SCORE = 2

    ## Data import
    print('============================================================')
    print('--------------------------------------------------')
    print('Load corpus')

    corpus = NewsCorpus(fdir_corpus=newspath.fdir_articles)
    DOCN = len(corpus)

    print(f'  | Corpus: {DOCN:,}')

    print('--------------------------------------------------')
    print('Load thesaurus')

    trash_word_list = newsio.read_thesaurus(fname_thesaurus=fname_trash_words)
    stoplist = newsio.read_thesaurus(fname_thesaurus=fname_stoplist)

    print(f'  | Trash words: {trash_word_list}')
    print(f'  | Stopwords: {stoplist}')

    ## Main
    print('============================================================')
    print('--------------------------------------------------')
    print('Preprocess text data')
    
    preprocess(corpus=corpus)