from elmo_search import *
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.filterwarnings("ignore")
import nltk
import pickle
from preprocessing import preproc
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf

import sys
# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
sys.path.append('simple_elmo')

from elmo_helpers import tokenize, get_elmo_vectors, load_elmo_embeddings

def crop_vec(vect, sent):
    """
    Crops dummy values

    :param vect: np.array, vector from ELMo
    :param sent: list of str, tokenized sentence
    :return: np.array

    """
    cropped_vector = vect[:len(sent), :]
    cropped_vector = np.mean(cropped_vector, axis=0)
    return cropped_vector

def prepare_elmo_query(query, batcher, sentence_character_ids, elmo_sentence_input):
    """ 
    Gets vector of query

    :param query: str
    :param batcher, sentence_character_ids, elmo_sentence_input: ELMo model
    
    :return: vector of query
    """
    query = preproc(query)
    q = [tokenize(query)] 
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        vector = crop_vec(get_elmo_vectors(sess, q, batcher,
                                           sentence_character_ids,
                                           elmo_sentence_input)[0], q[0])
    return vector


def search_tool_elmo(query, batcher, sentence_character_ids,
                     elmo_sentence_input, indexed): 
    indexed = pd.read_csv('elmo_index.csv', index_col=None)
    """
    Search query in corpus

    :param: query: str
    :param batcher, sentence_character_ids, elmo_sentence_input: ELMo model
    :param indexed: np.array, matrix of indexed corpus

    :return: list, sorted results
    """
    q = prepare_elmo_query(query, batcher, sentence_character_ids, 
                      elmo_sentence_input)

    result = {}
    q = np.nan_to_num(q)
    for i, doc_vector in enumerate(indexed):
        doc_vector = np.nan_to_num(doc_vector)
        #print(q)
        #score = cos_sim(q, doc_vector)
        score =  cosine_similarity(q.reshape(1, -1), doc_vector.reshape(1, -1))[0][0]
        if type(score) is np.float32:
            result[i] = score

    result = pd.DataFrame.from_dict(result, orient='index', columns=['val'])
    result['val'] = pd.to_numeric(result['val'])
    result = result.nlargest(10, 'val')
    return result
    #return sorted(result.items(), key=lambda x: x[1], reverse=True)[:10]

