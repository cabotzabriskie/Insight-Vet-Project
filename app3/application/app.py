from flask import Flask, render_template, request
import requests
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'\w+')
from gensim.models import KeyedVectors
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2


user = 'username'       
host = 'localhost'
dbname = 'VetRevs_db'
password = "password"
port     = '5432' 


db = create_engine('postgresql://{}:{}@{}:{}/{}'.format(user, password, host, port, dbname))

con = None
con = psycopg2.connect(database = dbname, user = user, host = host, password = password) 

w2vMod = KeyedVectors.load("data/w2vModel_keyvec.model", mmap='r')
def mean_vec (x):
    m = np.mean([w2vMod[token] for token in x], axis = 0)
    return m


def cos_all (x,y):
  cs = cosine_similarity(y.reshape(1,-1), x.reshape(1,-1))
  return cs

def ModelIt(fromUser, dat, model):
  doc1 = fromUser.strip().lower().split()
  remove = ['th','rd','st','']
  doc1P = [token for token in doc1 if (token not in remove and token in model.wv.vocab)]
  doc1PP = np.mean([model[token] for token in doc1P],axis=0)
  dat['sim'] = dat['wv'].apply(lambda x: cos_all(x,doc1PP))
  dat['sim2'] = dat['sim'].astype(float)
  winners = dat.nlargest(150, "sim2")

  winners.drop_duplicates(subset ="biz_name", 
                     keep = 'first', inplace = True)
  winners = winners.head(10)
  winners['sim2'] = winners['sim2'].round(4)
  
  return winners


#Initialize app
app = Flask(__name__, static_url_path='/static')

#Homepade
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/recommender', methods=['GET', 'POST'])
def recommender():
    return render_template('recommender.html')

@app.route('/recommendations', methods=['POST', 'GET'])
def recommendations():
    patient = request.form['statement1']
    
    try:
        sat = request.form['saturday']
    except:
        sat = "no_sat"
    try:
        sun = request.form['sunday']
    except:
        sun = "no_sun"
    
    if(patient is None):
        error = "Oh no, you didn't put in a review!"
        patient = "good"
        return render_template('recommender.html', test = "NoReview", link1 = "NoReview", rating1 = "NoReview", youmat1 = "NoReview", error = error)
    elif(len(patient) < 1):
        error = "Oh no, you didn't put in a review!"
        return render_template('recommender.html', test = "NoReview", link1 = "NoReview", rating1 = "NoReview", youmat1 = "NoReview", error = error)
    else:
        punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        no_punct = ""
        for char in patient:
            if char not in punctuations:
                no_punct = no_punct + char
        patient = no_punct
        query = "SELECT biz_name, tokens4, gmap_url, biz_url, biz_rating, sat_op, sun_op FROM chi_full_data_table;"
        print(query)
        query_results=pd.read_sql_query(query,con)
        query_results['tokens4'] = query_results['tokens4'].apply(tokenizer.tokenize)
        query_results['wv'] = query_results['tokens4'].apply(mean_vec)
        if sat == "saturday":
            query_results = query_results[query_results['sat_op'] == 0]
        if sun == "sunday":
            query_results = query_results[query_results['sun_op'] == 0]
        print(query_results)
        the_result = ModelIt(patient,query_results, w2vMod)
        the_result.reset_index
        print(the_result)
        test = the_result.iloc[0]['biz_name']
        link1 = the_result.iloc[0]['gmap_url']
        rating1 = the_result.iloc[0]['biz_rating']
        youmat1 = the_result.iloc[0]['sim2']*100
        test2 = the_result.iloc[1]['biz_name']
        link2 = the_result.iloc[1]['gmap_url']
        rating2 = the_result.iloc[1]['biz_rating']
        youmat2 = the_result.iloc[1]['sim2']*100
        test3 = the_result.iloc[2]['biz_name']
        link3 = the_result.iloc[2]['gmap_url']
        rating3 = the_result.iloc[2]['biz_rating']
        youmat3 = the_result.iloc[2]['sim2']*100
        test4 = the_result.iloc[3]['biz_name']
        link4 = the_result.iloc[3]['gmap_url']
        rating4 = the_result.iloc[3]['biz_rating']
        youmat4 = the_result.iloc[3]['sim2']*100
        test5 = the_result.iloc[4]['biz_name']
        link5 = the_result.iloc[4]['gmap_url']
        rating5 = the_result.iloc[4]['biz_rating']
        youmat5 = the_result.iloc[4]['sim2']*100
        return render_template('recommendations.html', test = test,test2 = test2,
                               test3 = test3,test4 = test4,test5 = test5, link1 = link1,
                               rating1 = rating1, youmat1 = youmat1, link2 = link2,
                               rating2 = rating2, youmat2 = youmat2,link3 = link3,
                               rating3 = rating3, youmat3 = youmat3,link4 = link4,
                               rating4 = rating4, youmat4 = youmat4,link5 = link5,
                               rating5 = rating5, youmat5 = youmat5,saturday = sat, sunday = sun)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
