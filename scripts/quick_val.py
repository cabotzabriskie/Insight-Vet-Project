#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 09:13:53 2019

@author: cabot
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'\w+')
from gensim.models import KeyedVectors



def cos_all (x,y):
    cs = cosine_similarity(y.reshape(1,-1), x.reshape(1,-1))
    return cs

def rmv_nonDict (x, rmv, vocab):
    w = [token for token in x if (token not in rmv and token in vocab)]
    return w

w2vMod = KeyedVectors.load("w2vModel_keyvec.model", mmap='r')

data = pd.read_csv("testval.csv")

data["tokens"] = data["review_text"].apply(tokenizer.tokenize)

data['tokens2'] = data['tokens'].apply(lambda x: ' '.join( [item for item in x if item not in stop_words]))

data["tokens3"] = data["tokens2"].apply(tokenizer.tokenize)

fromUser = "Mr buttons was having some issues the other night and we're new to the area and had no idea who to go to. We tried Dr Josephine and she was amazing in what ended up being a really terrible night. Unfortuneatly we had to put mr buttons to sleep because he was suffering but everyone on staff as so kind and made this difficult time bearable. Thank you so much."

doc1 = fromUser.strip().lower().split()
remove = ['th','rd','st','']
doc1P = [token for token in doc1 if (token not in remove and token in w2vMod.wv.vocab)]
doc1PP = np.mean([w2vMod[token] for token in doc1P],axis=0)
def mean_vec (x):
    m = np.mean([w2vMod[token] for token in x], axis = 0)
    return m

data['tokens4'] = data['tokens3'].apply(lambda x: rmv_nonDict(x, remove, w2vMod.wv.vocab))
data = data.dropna()
data['wv'] = data['tokens4'].apply(mean_vec)

data['sim'] = data['wv'].apply(lambda x: cos_all(x,doc1PP))
data['sim2'] = data['sim'].astype(float)

winners = data.nlargest(10,"sim2")

