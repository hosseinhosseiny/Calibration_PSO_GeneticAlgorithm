#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 16:43:54 2021

@author: hosseinhosseiny
"""
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel('RMSE_Tradition.xlsx')
x = data['cd'] 
y = data['rmse']

plt.figure()       
plt.plot(x,y, '-o')
plt.title('')
plt.xlabel('Drag Coefficient')
plt.ylabel('Root Mean Squared Error (RMSE)')
#plt.savefig('RMSE_Tradiotional.eps', dpi = 300, bbox_inches = "tight")