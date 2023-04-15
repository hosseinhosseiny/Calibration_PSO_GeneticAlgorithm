#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 13:28:10 2021

@author: hosseinhosseiny
"""
# -*- coding: utf-8 -*-

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

# data_tr['wse_sim_sc'] = ( data_tr['Calculatioin Result']- z_min)/(z_max -z_min)
# data_tr['wse_meas_sc'] = ( data_tr['Measured Values']- z_meas_min)/(z_meas_max -z_meas_min)

def _sc_n (n):
   sc_n =  ( n - n_min)/(n_max -n_min)
   return(sc_n)

def _wse (n):# n has the dimensions of (1,8)
    profile_st = model.predict([_sc_n(n)])
    ann_profile = (profile_st) * (z_max - z_min) + z_min
    return(ann_profile)
wse = np.vectorize(_wse)

def _fitness(x):# x must have the dimensions of (1,38)      
        err = mean_squared_error(x,wse_measT) 
        return round(err,6)
# fitness = np.vectorize(_fitness)   

def mutate(parents, fitness_function):
    n = int (len (parents))
    scores = fitness_function (parents)
    idx = scores > 0
    scores = scores [idx]
    parents = np.array(parents)[idx]
    ## resample parents with probabiilties proportionl to fitness
    ## then add some noise for random mutation
    childeren = np.random.choice(parents, size = n, p = scores / scores.sum())
    childeren = childeren + np.random.uniform(0.001,0.01, size = n) # add some noise to mutate
    return childeren.tolist()


## defining Genitic Algorithm
def GA(parents, fitness_function, popsize = 100, max_iter = 100):
    History = []
    ## initial parents; gen zero
    best_parent, best_fitness = _get_fittest_parent(parents, _fitness) # extracts fittest individual
    print ('generation {}| best fitness {}|curren fitness{}| current parent {}'.format(0,best_fitness, best_fitness, best_parent))
    
    ## plot initial parents
    x = np.linspace(start = 0.001, stop = 0.03, num = 30) # population range
    plt.plot(x, fitness_function(x))
    plt.scatter(parents, fitness_function(parents), marker = 'x')
    
    ## for each next generation
    for i in range (1, max_iter):
        parents = mutate(parents, fitness_function = fitness_function)
        
        curr_parent, curr_fitness = _get_fittest_parent (parents,fitness_function)
        
        # update best fitness value
        if curr_fitness > best_fitness:
            best_fitness = curr_fitness
            best_parent = curr_parent
            
        curr_parent, curr_fitness = _get_fittest_parent (parents,fitness_function)       
        
        if i % 10 == 0:
            print ('generation {}| best fitness {}|curren fitness{}| current parent {}'.format(i,best_fitness, curr_fitness, curr_parent))
        History.append((i,np.max(fitness_function(parents))))
            
            
    plt.scatter(parents,fitness_function(parents))
    plt.scatter(best_parent,fitness_function(best_parent), marker=".", c = 'b', s = 200)
    # plt.ylim(0,200)
    # plt.pause(0.09)
    # plt.ioff()# set the interactive mode off
    ##return best parents
    print('generation {}| best fitness {}| best parent {}'.format(i,best_fitness, best_parent))
    
    return best_parent, best_fitness, History
     
            
            
def _get_fittest_parent(parents, fitness):
    m = len(parents)
    fit = np.zeros((m,1))
    ws = np.zeros((m,38))
    for i in range(m):
        ws[i] = np.array(_wse([parents[i]]))
        fit[i]= np.array(_fitness ([ws[i]]))
    PFitness = list(zip(parents, fit))
    PFitness.sort( key = lambda x: x[1], reverse=False)
    best_parent, best_fitness = PFitness[0]
    return np.round(best_parent,4), np.round(best_fitness,4)            
            
x = np.linspace(start = 0.001, stop = 0.03, num = 30) # population range
init_pop = np.random.choice(x, size = (1,8))  
par = np.random.choice(x, size = (10,8)) 
par_mut = mutate(par[0], _fitness)

for i in range (10):
    print( _fitness(_wse([par[i]])))

par_fit = _get_fittest_parent(par, _fitness)







tes_pred = model.predict(init_pop)
tes_ann = _wse(init_pop)
tes_ann_T = tes_ann.transpose()
tes_ann_T_mse = mean_squared_error(tes_ann_T,wse_meas)
print(round(tes_ann_T_mse,6))


tes_fit = _fitness (tes_ann)


parent_, fitness_, history_ = GA(init_pop , _fitness)
            
            
x, y = list(zip(*history_))         
plt.plot(x,y)
plt.title('Maximum fitness')
plt.xlabel('Generation')
plt.ylabel('Fitness')
            
            
            
            
            
            
            