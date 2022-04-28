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

from object import NewsCorpus
from newsutil import NewsIO, NewsPath
newsio = NewsIO()
newspath = NewsPath()

import itertools
from tqdm import tqdm

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def prepare_docs_for_lda(corpus, fname, do):
    try:
        global SAMPLE_SIZE
    except:
        pass

    if do:
        docs_for_lda = {}
        for article in corpus.iter_sampling(n=SAMPLE_SIZE):
            docs_for_lda[article.id] = ' '.join(list(itertools.chain(*article.nouns_stop)))

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

def develop_lda_model(docs, num_topics, learning_decay, max_iter, batch_size):
    lda_model = LatentDirichletAllocation(n_components=num_topics,
                                          learning_decay=learning_decay,
                                          max_iter=max_iter,
                                          batch_size=batch_size,
                                          learning_method='online',
                                          evaluate_every=0,
                                          n_jobs=4,
                                          random_state=42,
                                          verbose=1
                                          )

    lda_model.fit_transform(docs)

    

    return lda_model

def gridsearch(fname_gs_result, do, **kwargs):
    if do:
        docs = kwargs.get('docs')
        parameters = kwargs.get('parameters')

        num_topics_list = parameters.get('num_topics')
        learning_decay_list = parameters.get('learning_decay')
        max_iter_list = parameters.get('max_iter')
        batch_size_list = parameters.get('batch_size')

        gs_result = {}
        candidates = itertools.product(*[num_topics_list, learning_decay_list, max_iter_list, batch_size_list])
        for candidate in tqdm(candidates):
            print('--------------------------------------------------')
            print(f'LDA modeling')
            print(f'  | candidate: {candidate}')
            num_topics, learning_decay, max_iter, batch_size = candidate
            fname_lda_model = f'lda/lda_{num_topics}_{learning_decay}_{max_iter}_{batch_size}.pk'

            try:
                lda_model = newsio.load(_type='model', fname_object=fname_lda_model)
            except FileNotFoundError:
                lda_model = develop_lda_model(docs, num_topics, learning_decay, max_iter, batch_size)
                newsio.save(_object=lda_model, _type='model', fname_object=fname_lda_model)

            perplexity = lda_model.perplexity(docs)
            loglikelihood = lda_model.score(docs)
            gs_result[fname_lda_model] = (perplexity, loglikelihood)
            print('--------------------------------------------------')
            print(f'LDA result')
            print(f'  | candidate: {candidate}')
            print(f'  | perplexity: {perplexity:,.03f}')
            print(f'  | loglikelihood: {loglikelihood:,.03f}')

        newsio.save(_object=gs_result, _type='result', fname_object=fname_gs_result)

    else:
        gs_result = newsio.load(_type='result', fname_object=fname_gs_result)

    return gs_result


if __name__ == '__main__':
    ## Parameters
    SAMPLE_SIZE = 10000

    DO_PREPARE_DOCS_FOR_LDA = False
    DO_VECTORIZE = False
    DO_GRIDSEARCH = True

    GS_PARAMETERS = {'num_topics': [5, 10, 15, 30, 50], #NUM_TOPICS
                     'learning_decay': [5e-1, 7e-1, 9e-1],
                     'max_iter': [10, 100, 500],
                     'batch_size': [64, 128, 256],
                    }


    ## Filenames
    fname_docs_for_lda = f'docs_{SAMPLE_SIZE}.pk'
    fname_docs_vectorized = f'docs_vectorized_{SAMPLE_SIZE}.pk'
    fname_gs_result = f'lda_gs_{SAMPLE_SIZE}.json'

    ## Data import
    print('============================================================')
    print('--------------------------------------------------')
    print('Load corpus')

    corpus = NewsCorpus(fdir_corpus=newspath.fdir_corpus)
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
    print('Gridsearch')

    gs_result = gridsearch(fname_gs_result=fname_gs_result, do=DO_GRIDSEARCH, docs=docs_vectorized, parameters=GS_PARAMETERS)

    print(gs_result)