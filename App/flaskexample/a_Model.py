#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 10:23:48 2019

@author: cabot
"""

import spacy
nlp = spacy.load('en')

def ModelIt(fromUser, dat):
  data2 = dat.sample(n = 1000, axis = 0)
  doc1 = nlp(fromUser)
  for index, row in data2.iterrows():
      data2.loc[index,'sim'] = doc1.similarity(nlp(data2.loc[index,'review_text']))
  winners = data2.nlargest(5, "sim")
  winners2 = winners.loc[:,"biz_name"]
  winners2 = winners2.reset_index()  
  return winners2.loc[0]['biz_name']