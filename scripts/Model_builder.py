#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 12:10:27 2019

@author: cabot
"""

import gensim
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'\w+')
from gensim.models import KeyedVectors

dbname = "VetRevs_db"
username = "username"
host     = 'localhost'
port     = '5432' 
password = "password"


#Import all reviews
con = None
con = psycopg2.connect(database = dbname, user = username, password = password, host =host, port = port)
sql_query = """
SELECT * FROM all_revs;
"""
data = pd.read_sql_query(sql_query,con)

data["tokens"] = data["review_text"].apply(tokenizer.tokenize)

data['tokens2'] = data['tokens'].apply(lambda x: ' '.join( [item for item in x if item not in stop_words]))

data["tokens3"] = data["tokens2"].apply(tokenizer.tokenize)

model = gensim.models.Word2Vec(
        data['tokens3'],
        size=150,
        window=10,
        min_count=2,
        workers=10)

model.train(data['tokens3'], total_examples=len(data['tokens3']), epochs=10)
word_vectors = model.wv
word_vectors.save("w2vModel_keyvec.model")

