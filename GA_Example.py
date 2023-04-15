# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 09:50:21 2021

@author: shossein
"""
import numpy as np
import matplotlib.pyplot as plt

def _fitness(x):
    "x= np.array(x)"
    if x > -11 and x < 11:
        y = (x**2+x) * np.cos(2*x) + x**2
        return round(y,6)
fitness = np.vectorize(_fitness)

x = np.linspace(start = -20, stop = 20, num = 200) # population range
plt.plot(x, fitness(x))       

def mutate(parents, fitness_function):
    n = int (len (parents))
    scores = fitness_function (parents)
    idx = scores > 0
    scores = scores [idx]
    parents = np.array(parents)[idx]
    ## resample parents with probabiilties proportionl to fitness
    ## then add some noise for random mutation
    childeren = np.random.choice(parents, size = n, p = scores / scores.sum())
    childeren = childeren + np.random.uniform(-0.51,0.51, size = n) # add some noise to mutate
    return childeren.tolist()


## defining Genitic Algorithm
def GA(parents, fitness_function, popsize = 100, max_iter = 100):
    History = []
    ## initial parents; gen zero
    best_parent, best_fitness = _get_fittest_parent(parents, fitness) # extracts fittest individual
    print ('generation {}| best fitness {}|curren fitness{}| current parent {}'.format(0,best_fitness, best_fitness, best_parent))
    
    ## plot initial parents
    x = np.linspace(start = -20, stop = 20, num = 200) # population range
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
    plt.ylim(0,200)
    plt.pause(0.09)
    plt.ioff()# set the interactive mode off
    ##return best parents
    print('generation {}| best fitness {}| best parent {}'.format(i,best_fitness, best_parent))
    
    return best_parent, best_fitness, History
     
            
            
def _get_fittest_parent(parents, fitness):
    _fitness = fitness (parents)
    PFitness = list(zip(parents, _fitness))
    PFitness.sort( key = lambda x: x[1], reverse=True)
    best_parent, best_fitness = PFitness[0]
    return round(best_parent,4), round(best_fitness,4)            
            
x = np.linspace(start = -20, stop = 20, num = 200) # population range
init_pop = np.random.uniform(low = -20, high = 20, size = 100)  

parent_, fitness_, history_ = GA(init_pop , fitness)
            
            
x, y = list(zip(*history_))         
plt.plot(x,y)
plt.title('Maximum fitness')
plt.xlabel('Generation')
plt.ylabel('Fitness')
            
            
            
            
            
            
            