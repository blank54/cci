'''
https://www.machinelearningplus.com/nlp/topic-modeling-python-sklearn-examples/
https://coredottoday.github.io/2018/09/17/%EB%AA%A8%EB%8D%B8-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0-%ED%8A%9C%EB%8B%9D/
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

import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel


def data_preparation(corpus, fname):
    try:
        lda_data = newsio.load(_type='data', fname_object=fname)
        docs_dict, id2word, docs_bow = lda_data
    except FileNotFoundError:
        docs_dict = {}
        for article in corpus.iter_sampling(n=SAMPLE_SIZE):
            docs_dict[article.id] = list(itertools.chain(*article.nouns_stop))

        id2word = corpora.Dictionary(docs_dict.values())
        docs_bow = [id2word.doc2bow(text) for text in docs_dict.values()]

        lda_data = [docs_dict, id2word, docs_bow]
        newsio.save(_object=lda_data, _type='data', fname_object=fname)

    return docs_dict, id2word, docs_bow

def develop_lda_model(docs_bow, id2word, num_topics, iterations, alpha, eta):
    lda_model = LdaModel(corpus=docs_bow,
                         id2word=id2word,
                         num_topics=num_topics,
                         iterations=iterations,
                         alpha=alpha,
                         eta=eta,)

    return lda_model

def calculate_coherence(lda_model, docs_dict):
    coherence_model = CoherenceModel(model=lda_model,
                                     texts=docs_dict.values())

    return coherence_model.get_coherence()

def gridsearch(fname_gs_result, do, **kwargs):
    if do:
        parameters = kwargs.get('parameters')
        docs_dict, id2word, docs_bow = data_preparation(corpus=corpus, fname=fname_lda_data)

        num_topics_list = parameters.get('num_topics')
        iterations_list = parameters.get('iterations')
        alpha_list = parameters.get('alpha')
        eta_list = parameters.get('eta')

        gs_result = {}
        candidates = itertools.product(*[num_topics_list, iterations_list, alpha_list, eta_list])
        for candidate in tqdm(candidates):
            print('\n--------------------------------------------------')
            print(f'LDA modeling')
            print(f'  | candidate: {candidate}')
            num_topics, iterations, alpha, eta = candidate
            fname_lda_model = f'lda/lda_{num_topics}_{iterations}_{alpha}_{eta}.pk'
            fname_coherence_model = f'coherence/coherence_{num_topics}_{iterations}_{alpha}_{eta}.pk'

            try:
                lda_model = newsio.load(_type='model', fname_object=fname_lda_model)
            except FileNotFoundError:
                lda_model = develop_lda_model(docs_bow, id2word, num_topics, iterations, alpha, eta)
                newsio.save(_object=lda_model, _type='model', fname_object=fname_lda_model)

            coherence_score = calculate_coherence(lda_model, docs_dict)
            gs_result[fname_lda_model] = coherence_score
            print('--------------------------------------------------')
            print(f'LDA result')
            print(f'  | candidate: {candidate}')
            print(f'  | coherence: {coherence_score:,.03f}')

        newsio.save(_object=gs_result, _type='result', fname_object=fname_gs_result)

    else:
        gs_result = newsio.load(_type='result', fname_object=fname_gs_result)

    return gs_result


if __name__ == '__main__':
    ## Parameters
    SAMPLE_SIZE = 10000

    DO_DATA_PREPARATION = True
    DO_GRIDSEARCH = True

    GS_PARAMETERS = {'num_topics': [5, 10, 15, 30, 50], #NUM_TOPICS
                     'iterations': [10, 100, 500],
                     'alpha': [0.1, 0.3, 0.5, 0.7],
                     'eta': [0.1, 0.3, 0.5, 0.7],
                    }

    ## Filenames
    fname_lda_data = f'docs_{SAMPLE_SIZE}.pk'
    fname_gs_result = f'lda_gs_{SAMPLE_SIZE}.json'

    ## Data import
    print('============================================================')
    print('--------------------------------------------------')
    print('Load corpus')

    corpus = NewsCorpus(fdir_corpus=newspath.fdir_corpus)
    print(f'  | Corpus: {len(corpus):,}')

    ## Topic modeling
    print('============================================================')
    print('Gridsearch')

    gs_result = gridsearch(fname_gs_result=fname_gs_result, do=DO_GRIDSEARCH, parameters=GS_PARAMETERS)

    print(gs_result)