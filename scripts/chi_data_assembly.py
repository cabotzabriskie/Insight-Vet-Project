#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 12:35:09 2019

@author: cabot
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
from gensim.models import KeyedVectors
import re


dbname = "VetRevs_db"
username = "username"
host     = 'localhost'
port     = '5432' 
password = "password"

engine = create_engine('postgresql://{}:{}@{}:{}/{}'.format(username, password, host, port, dbname))

con = None
con = psycopg2.connect(database = dbname, user = username, password = password, host =host, port = port)
sql_query = """
SELECT * FROM chi_google_id_data_table;
"""
data = pd.read_sql_query(sql_query,con)

data2 = pd.read_csv("Intermed-data/chi_gmap.csv")

data3 = data.merge(data2, how = "right",on = "biz_name")

data4 = data3[['biz_name','gmap_url']]

sql_query = """
SELECT * FROM chi_rmv_char_data_table;
"""
chi = pd.read_sql_query(sql_query,con)


drop = ['Bark N’ Bites',
'Temple of the Dog & Meow Lounge',
'PAWS Chicago Adoption Center',
'Kriser’s Natural Pet',
'Follow Your Nose',
'Chicago Canine Academy',
'Anything Is Pawzible',
'Doggy Style Pet Shop',
'The Anti-Cruelty Society',
'Tree House Humane Society',
'Bark Out Loud',
'Paws & Claws at Lincoln Square',
'DoGone Fun!',
'Meeow Chicago',
'Urban Pooch',
'Found Chicago Boarding and Training Center',
'Puptection Health & Nutrition Center',
'VIP’S Pet Hotels',
'Liz’s Pet Shop',
'Paradise Pet Salon & Spa TOO',
'Sit Means Sit',
'Chicago Cat Nannies',
'Earth Pups',
'Best Friends Pet Hotel',
'Pet-A-Cure',
'City Groomers',
'Chicago Canine Rescue',
'Bark Avenue Playcare Inc',
'Wag Your Tail',
'Pooch Hotel',
'Waggin’ Tails Shelter',
'Dog’s Life On Damen',
'Figo Pet Insurance',
'Pocket Puppies',
'Arfit',
'Spike’sBoutique Hotel for Dogs',
'Luv My Pet',
'Paradise 4 Paws',
'Pet Supplies Plus',
'Animal Care League',
'Pet Watchers',
'Paws to Heal',
'The Groomery',
'Cypress Woods Pet Crematory',
'The Animal Store',
'Pawsitive Petcare',
'CompanionAbility LLC',
'Home Pet Euthanasia of Chicago',
'Big Sky Dog Training',
'Doggy Daydream',
'Invisible Fence Brand',
'West Town Walkers',
'Right At Home Pet Service',
'Healthy Paws Pet Insurance & Foundation',
'Capricorn Dog Training',
'Dynamic Dogs Training & Behavior',
'Miller Robert K DVM',
'Chicago Veterinary Care',
'Pet Loss At Home - Home Euthanasia Vets',
'Sig P Hansen, DC',
'Emanoel Kotev DVM',
'Dona L Hernandez DVM',
"Brizgys Anthony E DVM",
"Pawblem Solved Pet Care",
"Chicago Veterinary House Calls",
"Patrick Mitchell, DVM",
"Amy and Aimee Care for Cats",
"My Paws & Claws",
"Yamaji P S DVM",
"Carlson Barbara DVM",
"Midwest Ragdolls",
"Doggy Love Camp",
"Ivan Veijic"]

chi = chi[~chi['biz_name'].isin(drop)]

chi2 = chi.merge(data4, how = "left", on = "biz_name")

chi2['sat'] = chi2['sat'].fillna("\n            Closed\n")
chi2['sat_op'] = np.where((chi2['sat'] == "\n            Closed\n ")|(chi2['sat'] == "\n            Closed\n"), 1, 0)
chi2['sun'] = chi2['sun'].fillna("\n            Closed\n")
chi2['sun_op'] = np.where((chi2['sun'] == "\n            Closed\n ")|(chi2['sun'] == "\n            Closed\n"), 1, 0)

word_vectors2 = KeyedVectors.load("w2vModel_keyvec.model", mmap='r')


from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'\w+')


chi2["tokens"] = chi2["review_text"].apply(tokenizer.tokenize)

chi2['tokens2'] = chi2['tokens'].apply(lambda x: ' '.join( [item for item in x if item not in stop_words]))

chi2["tokens3"] = chi2["tokens2"].apply(tokenizer.tokenize)

def rmv_nonDict (x,  vocab):
    w = [token for token in x if (token in vocab)]
    return w

def mean_vec (x):
    m = np.mean([word_vectors2[token] for token in x], axis = 0)
    return m

chi2['tokens4'] = chi2['tokens3'].apply(lambda x: rmv_nonDict(x, word_vectors2.vocab))


engine = create_engine('postgresql://{}:{}@{}:{}/{}'.format(username, password, host, port, dbname))

chi2.to_sql('chi_full_data_table', engine, if_exists = 'replace')
