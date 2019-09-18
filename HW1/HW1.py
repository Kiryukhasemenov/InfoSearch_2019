# -*- coding: utf-8 -*-

from pymorphy2 import MorphAnalyzer
from nltk.tokenize import WordPunctTokenizer
import re
import time
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
morph = MorphAnalyzer()

#ЧАСТЬ 1
#функция препроцессинга
def preproc(file, isFile=True):
    if isFile:
        with open(file, 'r', encoding='utf-8') as f:
            t = f.read()
    else:
        t = file
    t = re.sub(r'[A-Za-z0-9<>«»\.!\(\)?,;:\-\"\ufeff]', r'', t)
    text = WordPunctTokenizer().tokenize(t)
    preproc_text = ''
    for w in text:
        new_w = morph.parse(w)[0].normal_form + ' '
        preproc_text += new_w
    return preproc_text

#индексирование
def indexing(d):
    files = list(os.listdir(d))
    texts_words = []
    print('indexation of texts...')
    for file in files:
        t = file
        fpath = d + '/' + file
        ws = preproc(fpath)
        texts_words.append(ws)
        if len(texts_words) % 10 == 0:
            print(f'{len(texts_words)} texts done...')
            time.sleep(2)
    vec = CountVectorizer()
    X = vec.fit_transform(texts_words)
    df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names(), index=files)
    return df, vec

df_fr, vec = indexing('friends')

#функция поиска. Внутри - метрика релевантности (импортирована из sklearn)
def search(data, vec):
    q = input('Введите поисковой запрос: ')
    q = preproc(q, isFile=False)
    query = vec.transform([q])
    results = {}
    for index, row in data.iterrows():
        vector = row.as_matrix()
        cos_sim = cosine_similarity(vector.reshape(1, -1), query)
        cos_sim = np.asscalar(cos_sim)
        results[cos_sim] = index
    print('Вот 10 самых подходящих серий: ')
    idx = 1
    for key in sorted(results, reverse=True)[:10]:
        print(str(idx)+': '+results[key])
        idx += 1
    return None

search(df_fr, vec)

#ЧАСТЬ 2
#вспомогательная функция
def statistics_preproc(data, binary=False, noStopWords=False): 
    # параметр binary - приводит обратный индекс к бинарному виду (для каждого слова указывается только, встречается оно или нет),
    # параметр noStopWords - исключает из подсчета стоп-слова (на основе nltk.stopwords)
    df_transpose = data.transpose()
    if binary == True:
        df_transpose = df_transpose.applymap(lambda x: 1 if x > 0 else 0)
    df_transpose['sum'] = df_transpose.sum(axis=1)
    if noStopWords == True:
        df_transpose = df_transpose.drop(index=[stopword for stopword in russian_stopwords if stopword in df_transpose.index])
    return df_transpose
    
#задания a, b
def statistics_freq(data):
    freq_most = data.sort_values(by=['sum'], ascending=False).index[0]
    freq_least = data.sort_values(by=['sum'], ascending=False).index[-1]
    res = f'самое частотное слово: {freq_most}\nсамое редкое слово: {freq_least}'
    print(res)
    return res

#задание d
def monicaVSchandler(data):
    chendler_monika_stat = {}
    for i in range(1,8):
        num = str(i)+'x'
        num_cols = [col for col in data.columns if num in col]
        df_cols = data[num_cols]
        df_cols['sum_'] = df_cols.sum(axis=1)
        mon_chend = df_cols.loc[['моника','чендлера'], 'sum_']
        chendler_monika_stat[num] = list(mon_chend)
    chendler_monika_stat = pd.DataFrame.from_dict(chendler_monika_stat, orient='index', columns=['Моника','Чендлер'])
    print('самые популярные сезоны у двух героев:')
    print(chendler_monika_stat.idxmax())
    return None

#задание e
def most_popular(data):
    df_t_pop = data.loc[['фиби','моника','чендлера','рэйчел','джой','росс'], 'sum']
    sample = df_t_pop.to_dict()
    max_value = max(sample.items(), key=lambda x : x[1])
    print('наиболее популярный персонаж: ' + max_value[0].upper())
    return 'наиболее популярный персонаж: ', max_value

df_new = statistics_preproc(df_fr)
statistics_freq(df_new)
monicaVSchandler(df_new)
most_popular(df_new)
