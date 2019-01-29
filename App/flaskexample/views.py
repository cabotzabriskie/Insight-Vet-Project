#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 09:13:48 2019

@author: cabot
"""
from flask import render_template
from flask import request
from flaskexample import app
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2
import sys
from flaskexample.a_Model import ModelIt
# Python code to connect to Postgres
# You may need to modify this based on your OS, 
# as detailed in the postgres dev setup materials.
user = 'postgres' #add your Postgres username here      
host = 'localhost'
dbname = 'VetRevs_db'
password = "bobcoinc"
port     = '5432' 

#user = app.config["user"] #add your Postgres username here      
#host = app.config["host"] 
#dbname = app.config["dbname"] 
#password = app.config["password"] 
#port     = app.config["port"]  


db = create_engine('postgresql://{}:{}@{}:{}/{}'.format(user, password, host, port, dbname))

con = None
con = psycopg2.connect(database = dbname, user = user, host = host, password = password) #add your Postgres password here

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html",
       title = 'Home', user = { 'nickname': 'Miguel' },
       )

#@app.route('/db')
#def birth_page():
#    sql_query = """                                                                       
#                SELECT * FROM ny_rmv_char_data_table WHERE review_rating ='5';          
#                """
#    query_results = pd.read_sql_query(sql_query,con)
#    births = ""
#    for i in range(0,10):
#        births += query_results.iloc[i]['biz_name']
#        births += "<br>"
#    return births

#@app.route('/db_fancy')
#def cesareans_page_fancy():
#    sql_query = """
#               SELECT biz_name, review_name, biz_rating FROM ny_rmv_char_data_table WHERE review_rating ='5';
#                """
#    query_results=pd.read_sql_query(sql_query,con)
#    births = []
#    for i in range(0,query_results.shape[0]):
#        births.append(dict(index=query_results.iloc[i]['biz_name'], attendant=query_results.iloc[i]['review_name'], birth_month=query_results.iloc[i]['biz_rating']))
#    return render_template('cesareans.html',births=births)

@app.route('/input')
def cesareans_input():
    return render_template("input.html")

@app.route('/output')
def cesareans_output():
  #pull 'birth_month' from input field and store it
  patient = request.args.get('sentence')
    #just select the Cesareans  from the birth dtabase for the month that the user inputs
  query = "SELECT biz_name, review_text FROM ny_rmv_char_data_table WHERE review_rating ='5';"
  print(query)
  query_results=pd.read_sql_query(query,con)
  print(query_results)
  the_result = ModelIt(patient,query_results)  
  #births = []
  #for i in range(0,query_results.shape[0]):
   #   births.append(dict(biz_rating=query_results.iloc[i]['biz_rating'], biz_name=query_results.iloc[i]['biz_name']))
    #  the_result = ''
  return render_template("output.html", the_result = the_result)

