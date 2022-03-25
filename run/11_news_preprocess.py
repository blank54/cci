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
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from collections import defaultdict

from konlpy.tag import Komoran
komoran = Komoran()


def text_normalize(text):
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

def sent_normalize(sents, MIN_SENT_LEN, TRASH_SENT_SCORE):
    global trash_word_list

    concatenated_sents = concatenate_short_sent(sents=sents, MIN_SENT_LEN=MIN_SENT_LEN)

    output_sents = []
    for sent in concatenated_sents:
        trash_score = sum([1 if word in sent else 0 for word in trash_word_list])
        if trash_score < TRASH_SENT_SCORE:
            output_sents.append(sent)
        else:
            continue
    
    return output_sents

def remove_stopwords(sent, stoplist):
    return [w for w in sent if w not in stoplist]


if __name__ == '__main__':
    ## Filenames
    fname_trash_words = 'trashlist.txt'
    fname_stoplist = 'stoplist.txt'

    ## Parameters
    MIN_SENT_LEN = 3
    TRASH_SENT_SCORE = 2

    ## Data import
    print('============================================================')
    print('--------------------------------------------------')
    print('Load corpus')

    corpus = NewsCorpus(fdir_corpus=newspath.fdir_corpus)
    DOCN = len(corpus)

    print(f'  | Corpus: {DOCN:,}')

    print('--------------------------------------------------')
    print('Load thesaurus')

    trash_word_list = newsio.read_thesaurus(fname_thesaurus=fname_trash_words)
    stoplist = newsio.read_thesaurus(fname_thesaurus=fname_stoplist)

    print(f'  | Trash words: {trash_word_list}')
    print(f'  | Stopwords: {stoplist}')

    ## Preprocess
    print('============================================================')
    print('Preprocess text data')

    _start = datetime.now()
    for idx, doc in enumerate(corpus):
        ## Normalization
        normalized_text = text_normalize(text=doc.content)
        sents = parse_sent(text=normalized_text)
        normalized_sents = sent_normalize(sents=sents, MIN_SENT_LEN=MIN_SENT_LEN, TRASH_SENT_SCORE=TRASH_SENT_SCORE)
        doc.normalized_sents = deepcopy(normalized_sents)

        ## Tokenization, Stopword removal, and PoS tagging
        for sent in doc.normalized_sents:
            nouns = komoran.nouns(sent)
            doc.nouns = deepcopy(nouns)

            nouns_stop = remove_stopwords(sent=nouns, stoplist=stoplist)
            doc.nouns_stop = deepcopy(nouns_stop)

        ## Save corpus
        fpath_corpus = os.path.sep.join((newspath.fdir_corpus, doc.fname))
        with open(fpath_corpus, 'wb') as f:
            pk.dump(doc, f)

        ## Verbose
        if (idx+1) % 500 == 0:
            _end = datetime.now()
            running_time = _end - _start
            # running_time_form = datetime(running_time).strftime('%H:%M:%S')
            iter_time_avg = running_time/(idx+1)
            remaining_time = iter_time_avg * (DOCN-(idx+1))

            sys.stdout.write(f'\r  | {(idx+1):,}/{DOCN:,} [{str(running_time)}<{str(remaining_time)}, {str(iter_time_avg)}/iter]\t\t')