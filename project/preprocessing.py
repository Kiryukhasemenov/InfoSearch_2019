# -*- coding: utf-8 -*-
from pymorphy2 import MorphAnalyzer
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords 
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from math import log
from gensim.models.keyedvectors import KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import joblib
#import tensorflow as tf
import sys
import os



import logging
logging.basicConfig(filename='preprocessing.log', 
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%a, %d %b %Y %H:%M:%S',level=logging.INFO)

def dataframe_opening(): #достаем исходную таблицу, убираем лишние столбцы и переделываем все слова в начальные формы
    data = pd.read_csv("quora_question_pairs_rus.csv", index_col='Unnamed: 0')
    
    data = data.drop(['question2', 'is_duplicate'], axis=1)[:100] #эту строку оставить (убираем ненужные столбцы)

    data['question1'] = data['question1'].apply(lambda x: preproc(x)) #препроцессим (делаем леммы)
    data.to_csv('preprocessed_data.csv', index=True) #сохраняем в файле лемматизированные тексты
    return data

def preproc(text): #функция лемматизации и очистки от шелухи. Получает на вход одно предложение
    morph = MorphAnalyzer()
    text = re.sub(r'[A-Za-z0-9<>«»\.!\(\)?,;:\-\"]', r'', text)
    text = WordPunctTokenizer().tokenize(text)
    stopword_list = set(stopwords.words('russian'))
    
    preproc_text = ''
    for w in text:
        if w not in stopword_list:
            new_w = morph.parse(w)[0].normal_form + ' '
            preproc_text += new_w

    return preproc_text

def preproc_opening(): #открыть лемматизированный файл
    data = pd.read_csv("preprocessed_data.csv", index_col='Unnamed: 0')
    return data
    
#tf-idf vectorizer
def tf_idf_indexing(d): #получаем на вход список предложений, выдаем матрицу тф-идф
    vec = TfidfVectorizer()
    X = vec.fit_transform(d) 
    
    df_tfidf = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
    df_tfidf.to_csv('tf_idf_index.csv', index=False) 
    

    joblib.dump(vec, 'tf_idf_vectorizer.pkl') 
    return df_tfidf

def bm25_indexing(d, k=2, b=0.75): #получаем на вход список предложений, выдаем матрицу БМ25

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(d)
    term_freq_counts = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
    term_freq_counts['sum'] = term_freq_counts.sum(axis=1)
    
    tf_table = term_freq_counts.div(term_freq_counts['sum'], axis=0)
    tf_table = tf_table.fillna(0)    
    tf_table = tf_table.drop(['sum'], axis=1)
    
    bin_vectorizer = CountVectorizer(binary=True)
    bin_X = bin_vectorizer.fit_transform(d)
    bin_counts = pd.DataFrame(bin_X.toarray(), columns=bin_vectorizer.get_feature_names()) 
    word_counter_dict = {}
    
    for column in bin_counts.columns:
        col = bin_counts[column]
        sum_ = col.sum()
        word_counter_dict[column] = sum_
        
    inverse_counter = pd.DataFrame.from_dict(word_counter_dict, orient='index')
    inverse_counter = inverse_counter.transpose()
    
    #N = d.shape[0]
    N = len(d)
    idfs = {}
    
    for w in inverse_counter:
        idf = log((N - inverse_counter[w] + 0.5)/(inverse_counter[w] +0.5))
        idfs[w] = idf
        
    idf_table = pd.DataFrame.from_dict(idfs, orient='index')
    idf_table = idf_table.transpose()

    sums = term_freq_counts['sum']
    avg = term_freq_counts['sum'].mean()
    sums_normalized = sums.div(avg)

    #conversion_table = queries.mul(tf_table) #2
    conversion_table_numerator = tf_table.mul(k+1) #3
    coefficient = sums_normalized.mul(b) #4
    coefficient = coefficient.add(1-b) #5
    coefficient = coefficient.mul(k) # 6
    
    conversion_table_denominator = tf_table.mul(coefficient, axis=0) #7
    tf_factor = conversion_table_numerator.divide(conversion_table_denominator) #8
    tf_factor = tf_factor.fillna(0) #9
    n = tf_factor.shape[0]
    
    idf_table = pd.concat([idf_table]*n, ignore_index=True) #10 
    bm25_table = tf_factor.mul(idf_table, axis=1) #11
    bm25_table = bm25_table.fillna(0)
    bm25_table.to_csv('bm25_index.csv', index=False) #сохраняем в отдельный файл    
    return bm25_table

def getting_fasttext(filepath):
    fasttext_model = KeyedVectors.load(filepath)
    return fasttext_model



def sent_vectorizer(sent, model): #делаем вектора предложений в FastText
    if type(sent) != str:
        sent_vector = np.zeros((model.vector_size,))
        return sent_vector
    sent = sent.split()
    lemmas_vectors = np.zeros((len(sent), model.vector_size)) 
    for idx, lemma in enumerate(sent): 
        if lemma in model.vocab: 
            lemmas_vectors[idx] = model[lemma] 
    sent_vector = lemmas_vectors.mean(axis=0) 
    return sent_vector #потом записываем вектор этого предложения в наш большой индекс

def fasttext_indexing(d):
    model = getting_fasttext('fasttext/model.model')
    vectors_dict = {}
    
    for idx, row in d.iterrows():
        sent_vec = sent_vectorizer(row.question1, model)
        vectors_dict[idx] = sent_vec
        
    data = pd.DataFrame.from_dict(vectors_dict, orient='index')
    data.to_csv('fasttext_index.csv', index=False) 
    return data




def main():
    try:
        raw_df = dataframe_opening(use_both_cols=False)
        logging.info('made preprocessed dataframe')
        del(raw_df)
        preproc_df = preproc_opening()
        tf_idf_index = tf_idf_indexing(list(preproc_df.question1))
        logging.info('made tf-idf dataframe')
        del(tf_idf_index)
        bm25_index = bm25_indexing(list(preproc_df.question1))
        logging.info('made bm25 dataframe')
        del(bm25_index)
        fasttext_index = fasttext_indexing(preproc_df)
        logging.info('made fasttext dataframe')
        del(fasttext_index)
        #elmo_index = elmo_indexing(preproc_df)
        #logging.info('made ELMo dataframe')

    except Exception as e:
        logging.exception(repr(e) + ' while some function')


if __name__ == "__main__":
    main()

