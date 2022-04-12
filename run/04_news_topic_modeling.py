'''
https://www.machinelearningplus.com/nlp/topic-modeling-python-sklearn-examples/
'''

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
rootpath = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1])
sys.path.append(rootpath)

from object import NewsMonthlyCorpus
from newsutil import NewsIO, NewsPath
newsio = NewsIO()
newspath = NewsPath()

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichchletAllocation
from sklearn.model_selection import GridSearchCV


def prepare_docs_for_lda(corpus, fname, do):
    if do:
        docs_for_lda = {}
        for corpus_yearmonth in corpus.iter():
            for article in corpus_yearmonth:
                docs_for_lda[article.id] = article.nouns_stop

        newsio.save(_object=docs_for_lda, _type='data', fname_object=fname)

    else:
        docs_for_lda = newsio.load(_type='data', fname_object=fname)

    return docs_for_lda

def vectorize_docs(docs, fname, do):
    if do:
        vectorizer = CountVectorizer(analyzer='word', min_df=100)
        docs_vectorized = vectorizer.fit_transform(docs.values())

        newsio.save(_object=docs_vectorized, _type='data', fname_object=fname)

    else:
        docs_vectorized = newsio.load(_type='data', fname_object=fname)

    return docs_vectorized

def sparcity(data):
    data_dense = data.todense()
    return ((data_dense > 0).sum()/data_dense.size)*100


def init_lda_model(parameters):
    lda_model = LatentDirichchletAllocation(learning_method=parameters.get('learning_method'),
                                            random_state=parameters.get('random_state'),
                                            batch_size=parameters.get('batch_size'),
                                            evaluate_every=parameters.get('evaluate_every'),
                                            n_jobs=parameters.get('n_jobs'),
                                            )

    return lda_model

def gridsearch(fname, do, **kwargs):
    if do:
        lda_model = kwargs.get('lda_model')
        parameters = kwargs.get('parameters')
        docs_vectorized = kwargs.get('docs')

        gs_model = GridSearchCV(lda_model, param_grid=parameters)
        gs_model.fit(docs_vectorized)

        newsio.save(_object=gs_model, _type='model', fname_object=fname)

    else:
        gs_model = newsio.load(_type='model', fname_object=fname)

    return gs_model

def show_best_model(gs_model):
    



if __name__ == '__main__':
    ## Parameters
    CORPUS_START = '200501'
    CORPUS_END = '202112'

    DO_PREPARE_DOCS_FOR_LDA = False
    DO_VECTORIZE = True
    DO_GRIDSEARCH = True

    LDA_PARAMETERS = {'learning_method': 'online',
                      'random_state': 42,
                      'batch_size': 128,
                      'evaluate_every': -1,
                      'n_jobs': -1,
                      }
    GS_PARAMETERS = {'n_components': [10, 15, 20, 25, 30], #NUM_TOPICS
                     'learning_decay': [0.5, 0.7, 0.9],
                     'max_iter': [10, 100, 500],
                    }


    ## Filenames
    fname_docs_for_lda = f'docs_{CORPUS_START}_{CORPUS_END}.pk'
    fname_docs_vectorized = f'docs_vectorized_{CORPUS_START}_{CORPUS_END}.pk'
    fname_gs_model = f'lda_gs_model_{CORPUS_START}_{CORPUS_END}.pk'

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

    print('--------------------------------------------------')
    print('Vectorization')

    docs_vectorized = vectorize_docs(docs=docs_for_lda, fname=fname_docs_vectorized, do=DO_VECTORIZE)
    data_sparcity = sparcity(data=docs_vectorized)
    print(f'  | Shape: {docs_vectorized.shape}')
    print(f'  | Sparcity: {data_sparcity:,.03f}')

    ## Topic modeling
    print('============================================================')
    print('--------------------------------------------------')
    print('Init LDA model')

    lda_model = init_lda_model(parameters=LDA_PARAMETERS)
    print(f'  | {lda_model}')

    print('--------------------------------------------------')
    print('Gridsearch')

    gs_model = gridsearch(fname=fname_gs_model, do=DO_GRIDSEARCH, lda_model=lda_model, parameters=GS_PARAMETERS, docs=docs_vectorized)

    print('--------------------------------------------------')
    print('Best model')

    lda_model_best = gs_model.best_estimator_
    print(f'  | Parameters: {gs_model.best_params_}')
    print(f'  | Log likelihood score: {gs_model.best_score_:,.03f}')
    print(f'  | Perplexity: {lda_model_best.perplexity(docs_vectorized)}')

    # print('--------------------------------------------------')
    # print('Performance evaluation')

    # 