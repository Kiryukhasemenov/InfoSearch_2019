# -*- coding: utf-8 -*-
import logging
logging.basicConfig(filename='application.log', 
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%a, %d %b %Y %H:%M:%S',level=logging.INFO)
from flask import Flask
from flask import url_for, render_template, request, redirect
from search_and_eval import *
import sys
# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
sys.path.append('simple_elmo')

from elmo_helpers import tokenize, get_elmo_vectors, load_elmo_embeddings

app = Flask(__name__)

@app.route('/search')
def dosearch():
    if request.args:
        try:
            query=request.args['query']
            method=request.args['method']
            logging.info(f'Query: {query}, Method: {method}')
            result=search(query, method)
            res = [val for key, val in result.items()]
            logging.info(f'Result: {res}')
            return render_template('results.html', query=query, method=method, result=result)
        except Exception as e:
            logging.exception(f'Error with query {query} and model {method}:\n\n repr(e)')

            
    return render_template('search.html')

@app.route('/')
def index():
    urls = {'Привет! Поиск тут: ': url_for('dosearch'),}
    return render_template('index.html', urls=urls)



if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)
