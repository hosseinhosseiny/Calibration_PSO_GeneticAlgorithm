#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 12:41:07 2021

@author: hosseinhosseiny
"""
# %import matlab.engine
# %eng = matlab.engine.start_matlab()



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
import pickle
import os
from tensorflow.keras.models import Sequential
from sklearn.metrics import mean_squared_error
import collections
from collections import Counter 
  

#-------------setting randomness condition
seed = 6
np.random.seed = seed
random.seed(seed)
tf.random.set_seed(seed)
#-----------------------loading the model and data
model = keras.models.load_model('GA_ANN')
data_dist = pd.read_excel('dist.xlsx')
dist = data_dist['Distance']
distT = pd.DataFrame(dist).T
wse_meas = data_dist['Measured Values']
wse_measT = pd.DataFrame(wse_meas).T

pickle_in = open('z_max', 'rb')
z_max = pickle.load(pickle_in)

pickle_in = open('z_min', 'rb')
z_min = pickle.load(pickle_in)

pickle_in = open('n_max', 'rb')
n_max = pickle.load(pickle_in)
#------- load mask no data
pickle_in = open('n_min', 'rb')
n_min = pickle.load(pickle_in)
# -----loading depth 


# pop_num = 100
# iter_num = 100
# x = np.linspace(start = 0.001, stop = 0.03, num = 30) # population range 
# init_pop = np.random.choice(x, size = (pop_num,8)) 
#--------
def _sc_n (n):
   sc_n =  ( n - n_min)/(n_max -n_min)
   return(sc_n)

def _wse (n):# n has the dimensions of (1,8)
    profile_st = model.predict([_sc_n(n)])
    ann_profile = (profile_st) * (z_max - z_min) + z_min
    return(ann_profile)

def _fitness(x):# x must have the dimensions of (1,38)      
        RMSE = mean_squared_error(x,wse_measT,squared=False) 
        return round(RMSE,6)
 

# eng.PSO_iRIC(nargout = 0)


n = [0.0022, 0.0113, 0.0051, 0.0091, 0.0081, 0.009, 0.0079, 0.0057]
wse_n = _wse([n])
rmse_n = _fitness(wse_n)













