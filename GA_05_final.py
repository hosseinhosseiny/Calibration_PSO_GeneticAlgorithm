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


pop_num = 100
iter_num = 100
x = np.linspace(start = 0.001, stop = 0.03, num = 30) # population range 
init_pop = np.random.choice(x, size = (pop_num,8)) 
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
 
def most_frequent(List): 
    occurence_count = Counter(List) 
    return occurence_count.most_common(1)[0][0] 
    
 
def mutate(parents, fitness_function):
    n = len(parents) # row #
    m = len(parents[0]) # column #
    scores = np.zeros((n))
    childeren = np.zeros((n,m))
    for j in range(n):
        scores[j] = np.array(fitness_function(_wse([parents[j]])))
    prob1 =  ((1/scores) / ((1/scores).sum()) )
    ind_bst_par = np.where(prob1 == max(prob1))[0][0]
  
    childeren[n-1] = parents[(ind_bst_par)]  #[0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008]
    # print(ind_bst_par)
    # print('probability=:\n', prob1)
    index =  np.random.choice(n,size = n-1, p = prob1)
    # print('index',index)
    childeren[:-1] = parents[(index)]
    #-- adding noise
    xx = np.linspace(start = 0.001, stop = 0.006, num = 5) # random range
    # childeren[:-1] = childeren[:-1] + np.random.choice(xx, size = (n-1,m))
    childeren2 = np.zeros((n,m))
    childeren2[n-1] = parents[(ind_bst_par)] 
    for i in range (0,n-1):      
        col_ind = np.random.choice(range(1,m-1))
        # print('colum index\n',col_ind)
        childeren2[i][:col_ind] = childeren[i+1][-col_ind:]
    # for i in range (1,n-1,2):
    #     col_ind = np.random.choice(range(1,m-1))
    #     print('colum index',col_ind)
        childeren2[i][col_ind:] = childeren[i-1][:-col_ind]
    for i in range(n):    
        col_ind = np.random.choice(m)
        row_ind= np.random.choice(n-1)
        # num = np.random.choice(x)
        childeren2[row_ind][col_ind] = childeren2[row_ind][col_ind] + np.random.choice(xx)
    # print('parents\n',parents)
    # print('childeren\n',childeren)  
    # print('childeren2\n',childeren2) 
    return childeren2


## defining Genitic Algorithm
def GA(parents, fitness_function, popsize = pop_num, max_iter = iter_num):
    History = []
    ## initial parents; gen zero
    best_parent, best_fitness = _get_fittest_parent(parents, _fitness) # extracts fittest individual
    print ('generation {}| best RMSE {}|curren RMSE {}| current parent {}'.format(0,best_fitness, best_fitness, best_parent))

    ## for each next generation

    for i in range (1,max_iter):
        parents = mutate(parents, fitness_function = fitness_function)
       
        curr_parent, curr_fitness = _get_fittest_parent (parents,fitness_function)
        # print(i)
        # update best fitness value
        if curr_fitness < best_fitness:
            best_fitness = curr_fitness
            best_parent = curr_parent
            
        # curr_parent, curr_fitness = _get_fittest_parent (parents,fitness_function)       
        
        if i % 10 == 0:
            print ('generation {}| best RMSE {}|current RMSE{}| current parent {}'.format(i,best_fitness, curr_fitness, curr_parent))
            # for j in range (n):
        History.append((i,fitness_function(_wse([best_parent]))))
       
    print('generation {}| best RMSE {}| best parent {}'.format(i,best_fitness, best_parent))
    
    return best_parent, best_fitness, History
     
            
def _get_fittest_parent(parents, fitness):
    m = len(parents)
    fit = np.zeros((m,1))
    ws = np.zeros((m,38))
    for i in range(m):
        ws[i] = np.array(_wse([parents[i]]))
        fit[i]= np.array(_fitness ([ws[i]]))
    PFitness = list(zip(parents, fit))
    PFitness.sort( key = lambda x: x[1], reverse=False)# False is correct
    best_parent, best_fitness = PFitness[0]
    return np.round(best_parent,4), np.round(best_fitness,4)            
            


parent_, fitness_, history_ = GA(init_pop , _fitness)
        
            
x, y = list(zip(*history_)) 
 
plt.figure()       
plt.plot(x,y)

plt.minorticks_on()
plt.grid(b=True, which='both',axis='y', color='grey', linestyle='--')
plt.xticks(np.arange(0, 100, 10))
plt.yticks(np.arange(0.02, 0.15, 0.02))
plt.title('')
plt.xlabel('Iteration (Generation)')
plt.ylabel('Best Cost')

plt.savefig('GA_Perf_Final', dpi = 300, bbox_inches = "tight")
            
yy = (_wse([parent_]) ).flatten()            
plt.figure()   
plt.plot(distT.to_numpy().flatten(),yy,"bo",label = 'GA Best n') 
plt.plot(distT.to_numpy().flatten(),wse_measT.to_numpy().flatten(),"go",label = 'Measured') 
plt.title('')
plt.legend()
plt.xlabel('Distance-m')
plt.ylabel('Elevation-m')            
plt.savefig('GA_Final_WS profile', dpi = 300, bbox_inches = "tight")       
            
# # save data
# pickle_out = open('GA_History', 'wb')
# pickle.dump([parent_, fitness_, history_], pickle_out)
# pickle_out.close()            



# plot with matlab- didnt work
# import matlab.engine as eng
# eng.figure()
# # plot (BestCosts, 'LineWidth',2)
# eng.semilogy (y, 'LineWidth',2)
# eng.xlabel('Iteration')
# eng.ylabel('Best Cost')
# # eng.grid on







         