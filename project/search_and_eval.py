# -*- coding: utf-8 -*-

from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import preproc, sent_vectorizer, getting_fasttext
from elmo_search import *
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.filterwarnings("ignore")
import nltk
#nltk.download('stopwords')
w2v = getting_fasttext('fasttext/model.model')
import sys
#the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
sys.path.append('simple_elmo')
from elmo_helpers import tokenize, get_elmo_vectors, load_elmo_embeddings
elmo_path = 'ELMO'

batcher, sentence_character_ids, elmo_sentence_input = load_elmo_embeddings(elmo_path)

def query_preprocessing(query, model):
    query_preprocessed = preproc(query)
    return query_preprocessed

def metric(query, model):
    if model == 'TF-IDF':
        df = pd.read_csv('tf_idf_index.csv', index_col=None)
        vectorizer = joblib.load('tf_idf_vectorizer.pkl')
        query_tfidf = vectorizer.transform([query])
        query_tfidf = pd.DataFrame(query_tfidf.toarray(), columns=vectorizer.get_feature_names())
        metric_value = calc_metric(query_tfidf, df)


    elif model == 'BM25':
        df = pd.read_csv('bm25_index.csv', index_col=None)
        query = query.split(' ')
        
        lemmas_list = list(df.columns)
        query_bm25 = {}
        for lemma in lemmas_list:
            if lemma in query:
                query_bm25[lemma] = [1]
            else:
                query_bm25[lemma] = [0]

        query_bm25 = pd.DataFrame.from_dict(query_bm25)
        
        metric_value = calc_metric(query_bm25, df)

    elif model == 'FastText':
        df = pd.read_csv('fasttext_index.csv', index_col=None)
        sent_vector = sent_vectorizer(query, w2v)

        query_fasttext = np.asarray(sent_vector).reshape(1, -1)
        metric_value = calc_metric(query_fasttext, df)

    else:
        with open('Indexed_ELMO.pickle', 'rb') as f:
            indexed, id_to_text, query_to_id = pickle.load(f)
        
        #query = prepare_elmo_query(query, batcher, sentence_character_ids, elmo_sentence_input)
        metric_value = search_tool_elmo(query, batcher, sentence_character_ids, elmo_sentence_input, indexed)
        
        #metric_value = 'NAN'
    return metric_value

def calc_metric(query, data):
    cos_sim = data.apply(lambda row: cosine_similarity(row.values.reshape(1, -1), query)[0][0], axis=1)
    cos_sim = pd.DataFrame(cos_sim, columns=['val'])
    best_cos_sim = cos_sim.nlargest(10, 'val')
    return best_cos_sim
   
    
def get_ultimate_sentences(res):
    print(type(res))
    print(res)
    ultimate_df = pd.read_csv("quora_question_pairs_rus.csv", index_col='Unnamed: 0')
    res['dupl'] = res.index
    res['sentence'] = res['dupl'].apply(lambda x: ultimate_df.loc[x,'question1'])
    res.drop(['dupl'], axis=1)
    
    res_dict = {}
    for idx, row in res.iterrows():
        res_dict[idx] = [row.sentence, row.val]

    return res_dict

def search(query, model):
    query = query_preprocessing(query, model)
    evaluation_df = metric(query, model)
    final_df = get_ultimate_sentences(evaluation_df)
    return final_df

#def main():
#    search(input('Введите запрос: '), input('Введите модель: '))

#if __name__ == "__main__":
#    main()
