#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 16:43:54 2021

@author: hosseinhosseiny
"""
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel('GA_PSO_Calibrated WSProfiles.xlsx')
x = data[' Stream_wise Distance']

y_pso = data['Calculatioin Result_PSO'] 
y_ga = data['Calculatioin Result_GA']
y_usgs = data['Calculatioin Result_usgs']  
y_meas = data['Measured Values']

plt.figure(figsize=(18,10) )    
plt.plot(x,y_pso, '-o', markerfacecolor = 'none', markersize= 15, label='PSO (RMSE=0.036 m)')
plt.plot(x,y_ga, '-.2', markersize= 10, label = 'GA (RMSE=0.038 m)')
plt.plot(x,y_usgs, '-.+', markersize= 10, label = 'Trial and Error (RMSE=0.054 m)')  
plt.plot(x,y_meas, 's', markerfacecolor = 'none', markersize= 15, label = 'Measured')
plt.legend(fontsize=(25))
plt.xticks(fontsize=(20))
plt.yticks(fontsize=(20))
plt.xlabel('Distance (m)',fontsize=(25) )
plt.ylabel('Elevation (m)',fontsize=(25))


plt.savefig('WSE_COmpare_GA_PSO_Unit', dpi = 300, bbox_inches = "tight")