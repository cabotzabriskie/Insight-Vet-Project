#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 09:10:32 2019

@author: cabot
"""

from flask import Flask
app = Flask(__name__)#, instance_relative_config=True)
from flaskexample import views
#app.config.from_object('config')
#app.config.from_pyfile('config.py')