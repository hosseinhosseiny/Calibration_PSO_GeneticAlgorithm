#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 08:18:31 2021

@author: hosseinhosseiny
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
import matplotlib.pyplot as plt
import pickle
seed = 6
np.random.seed = seed
random.seed(seed)
tf.random.set_seed(seed)
import os
from tensorflow.keras.models import Sequential

data = []

path = os.getcwd()
files = os.listdir(path)
files_xls = [f for f in files if f[-4:] == 'xlsx']
df = pd.DataFrame()
for f in files_xls:
    data = pd.read_excel(f, 'Sheet1')
    df = df.append(data)   
n_max = max(df['n'])
n_min = min(df['n'])

z_max = max(df['Calculatioin Result'])
z_min = min(df['Calculatioin Result'])

z_meas_max = max(df['Measured Values'])
z_meas_min = min(df['Measured Values'])

df['n_sc'] = ( df['n']- n_min)/(n_max -n_min)
df['wse_sim_sc'] = ( df['Calculatioin Result']- z_min)/(z_max -z_min)
df['wse_meas_sc'] = ( df['Measured Values']- z_meas_min)/(z_meas_max -z_meas_min)

#-------------------- Data Test
data_test = pd.read_excel("test_total.xlsx")
data_test['n1_sc'] = (data_test['n1']- n_min)/(n_max -n_min)
data_test['n2_sc'] = (data_test['n2']- n_min)/(n_max -n_min)
data_test['n3_sc'] = (data_test['n3']- n_min)/(n_max -n_min)
data_test['n4_sc'] = (data_test['n4']- n_min)/(n_max -n_min)
data_test['n5_sc'] = (data_test['n5']- n_min)/(n_max -n_min)
data_test['n6_sc'] = (data_test['n6']- n_min)/(n_max -n_min)
data_test['n7_sc'] = (data_test['n7']- n_min)/(n_max -n_min)
data_test['n8_sc'] = (data_test['n8']- n_min)/(n_max -n_min)

data_test['wse_sim_sc'] = ( data_test['Calculatioin Result']- z_min)/(z_max -z_min)
data_test['wse_meas_sc'] = ( data_test['Measured Values']- z_meas_min)/(z_meas_max -z_meas_min)

X_test = data_test[['n1_sc', 'n2_sc', 'n3_sc', 'n4_sc', 'n5_sc', 'n6_sc', 'n7_sc', 'n8_sc' ]]
y_test = data_test['wse_sim_sc']

#-------------------Data Train
n1 = df[df['location'] == 'n1']   
n2 = df[df['location'] == 'n2']    
n3 = df[df['location'] == 'n3']   
n4 = df[df['location'] == 'n4']     
n5 = df[df['location'] == 'n5']   
n6 = df[df['location'] == 'n6']   
n7 = df[df['location'] == 'n7']   
n8 = df[df['location'] == 'n8']   

n_val = (df['n']).unique()
n_val = n_val[~np.isnan(n_val)]

n_list = [n1, n2, n3, n4, n5, n6, n7, n8]

data_y = pd.DataFrame(np.zeros((len(n_val) * 8,38))) # 19 n values * 8 regions = 152|| 38 is the number of the points in WAS profile
data_x = pd.DataFrame(np.zeros((len(n_val) * 8,38))) # 19 n values * 8 regions = 152|| 38 is the number of the points in WAS profile

l = 0
for n in n_list:
    dat = n
    for j in range(0,722,38):
        # print(j)
        tr = (dat['wse_sim_sc'][j:j+38])
        tr_n = (dat['n'][j:j+38])
        
        data_y.loc[l][:] = tr
        data_x.loc[l][:] = tr_n
        # print(l)
        l = l+1
data_x = data_x.iloc[:,:8]
 
X_train = data_x
y_train =  data_y
  
#-------------------Define Sequential model with 3 layers
model = keras.Sequential()
model.add(layers.Dense(50, activation="relu", input_shape=(8,)))
# model.add(layers.Dense(76, activation="relu"))
model.add(layers.Dense(38))
model.summary()

# Compile model
learning_rate= 0.5
# model= tf.keras.Model(inputs=[inputs], outputs=[outputs])
opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
model.compile(optimizer=opt,
              loss='mean_squared_error',
              metrics=['mse', 'mae'])

BATCH_SIZE = 200
EPOCHS = 3

# Train model
results = model.fit(X_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_split = 0.2,
          verbose=1,
          validation_data=(X_test, y_test))
          
score = model.evaluate(X_test, y_test, verbose=0)
print('Test mean_squared_error:', score[0])
print('Test mae:', score[2])

wse_ann_sc = model.predict(X_test)
wse_ann = (wse_ann_sc) * (z_max - z_min) + z_min

wse1 = wse_ann[0]
wse1_sim = data_test.iloc[0:38, 0]

wse2 = wse_ann[38]
wse2_sim = data_test.iloc[38:76, 0]


wse3 = wse_ann[76]
wse3_sim = data_test.iloc[76:114, 0]


wse4 = wse_ann[114]
wse4_sim = data_test.iloc[114:152, 0]


plt.figure()

































#----------------------------------------- plotting the accuracy history
def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    
    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], '-.g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
    plt.xlim(xmin=0)
    # plt.ylim(ymin=0.065, ymax=0.1)
    # plt.title('Loss-n={}, batch={}, lr={}'.format(X_train.shape, batch_size, learning_rate))
    plt.xlabel('Epochs')
    plt.ylabel('Loss (mean squared error)')
    plt.legend()
   
#import above function and pass the parameter used while training    
plot_history(results)    
plt.savefig('Final_Performance1', dpi = 300)
# plt.savefig('n={}'.format(X_train.shape))


#-------------save the model, data, and results

with open('trainHistoryDict', 'wb') as file_pi:
        pickle.dump(results.history, file_pi)
        
model.save('GA_ANN')# 

# save data
pickle_out = open('X_train_final', 'wb')
pickle.dump(X_train, pickle_out)
pickle_out.close()

pickle_out = open('y_train_final', 'wb')
pickle.dump(y_train, pickle_out)
pickle_out.close()

pickle_out = open('X_test', 'wb')
pickle.dump(X_test, pickle_out)
pickle_out.close()

pickle_out = open('y_test', 'wb')
pickle.dump(y_test, pickle_out)
pickle_out.close()

pickle_out = open('n_max', 'wb')
pickle.dump(n_max, pickle_out)
pickle_out.close()

pickle_out = open('n_min', 'wb')
pickle.dump(n_min, pickle_out)
pickle_out.close()


pickle_out = open('z_max', 'wb')
pickle.dump(z_max, pickle_out)
pickle_out.close()

pickle_out = open('z_min', 'wb')
pickle.dump(z_min, pickle_out)
pickle_out.close()

pickle_out = open('z_meas_max', 'wb')
pickle.dump(z_meas_max, pickle_out)
pickle_out.close()

pickle_out = open('z_meas_min', 'wb')
pickle.dump(z_meas_min, pickle_out)
pickle_out.close()

pickle_out = open('df', 'wb')
pickle.dump(df, pickle_out)
pickle_out.close()








