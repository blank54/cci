#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
rootpath = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1])
sys.path.append(rootpath)

from object import NewsMonthlyCorpus, NewsTopicModel
from newsutil import NewsIO, NewsPath
newsio = NewsIO()
newspath = NewsPath()


def prepare_docs_for_lda(corpus, fname, do):
    if do:
        docs_for_lda = {}
        for corpus_yearmonth in corpus.iter():
            for article in corpus_yearmonth:
                docs_for_lda[article.id] = article.content.split()

        newsio.save(_object=docs_for_lda, _type='data', fname_object=fname)

    else:
        docs_for_lda = newsio.load(_type='data', fname_object=fname)

    return docs_for_lda

def develop_lda_model(docs, fname, do):
    global NUM_TOPICS, LDA_PARAMETERS

    if do:
        print('--------------------------------------------------')
        print('Init LDA model')

        lda_model = NewsTopicModel(docs=docs, num_topics=NUM_TOPICS)
        lda_model.fit()

        print('--------------------------------------------------')
        print('Save LDA model')

        newsio.save(_object=lda_model, _type='model', fname_object=fname)

    else:
        print('--------------------------------------------------')
        print('Load LDA model')

        lda_model = newsio.load(_type='model', fname_object=fname)

    print('--------------------------------------------------')
    print('Coherence of LDA model')
    print(f'  | coherence: {lda_model.coherence:,}')

    return lda_model


if __name__ == '__main__':
    ## Parameters
    CORPUS_START = '202101'
    CORPUS_END = '202103'

    DO_PREPARE_DOCS_FOR_LDA = False
    DO_TOPIC_MODELING = True

    NUM_TOPICS = 10
    LDA_PARAMETERS = {'iterations': 3,
                      'alpha': 0.2,
                      'eta': 0.1,
                      }

    ## Filenames
    fname_docs_for_lda = f'docs_{CORPUS_START}_{CORPUS_END}.pk'
    fname_lda_model = f'lda_model_{CORPUS_START}_{CORPUS_END}_{str(NUM_TOPICS)}.pk'

    ## Data import
    print('============================================================')
    print('--------------------------------------------------')
    print('Load corpus')

    corpus = NewsMonthlyCorpus(fdir_corpus=newspath.fdir_corpus_monthly, start=CORPUS_START, end=CORPUS_END)
    DOCN = len(corpus)

    print(f'  | Corpus: {DOCN:,}')

    print('--------------------------------------------------')
    print('Docs for LDA')

    docs_for_lda = prepare_docs_for_lda(corpus=corpus, fname=fname_docs_for_lda, do=DO_PREPARE_DOCS_FOR_LDA)

    print(f'  | {len(docs_for_lda):,} articles')

    ## Topic modeling
    print('============================================================')

    lda_model = develop_lda_model(docs=docs_for_lda, fname=fname_lda_model, do=DO_TOPIC_MODELING)