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
# df_tr_final


pickle_in = open('df_tr_final', 'rb')
data_tr= pickle.load(pickle_in)

path = os.getcwd()
files = os.listdir(path)

colum = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8' ]
colum_sc = ['n1_sc', 'n2_sc', 'n3_sc', 'n4_sc', 'n5_sc', 'n6_sc', 'n7_sc', 'n8_sc' ]

n_val = (data_tr['n']).unique()
# n_val = n_val[~np.isnan(n_val)]
wse_points = 38
reg_num = 8 # number of the regions in hydraulic model
n_num = len(n_val)

   
n_max = max(n_val)
n_min = min(n_val)

z_max = max(data_tr['Calculatioin Result'])
z_min = min(data_tr['Calculatioin Result'])

z_meas_max = max(data_tr['Measured Values'])
z_meas_min = min(data_tr['Measured Values'])

data_tr['n_sc'] = ( data_tr['n']- n_min)/(n_max -n_min)

data_tr['wse_sim_sc'] = ( data_tr['Calculatioin Result']- z_min)/(z_max -z_min)
data_tr['wse_meas_sc'] = ( data_tr['Measured Values']- z_meas_min)/(z_meas_max -z_meas_min)

#-------------------- Data Test
data_test = pd.read_excel("TEST_Total.xlsx")

for n, nsc in zip (colum, colum_sc):
    print(n, nsc)
    data_test[nsc] = ( data_test[n]- n_min)/(n_max -n_min)

data_test['wse_sim_sc'] = ( data_test['Calculatioin Result']- z_min)/(z_max -z_min)
data_test['wse_meas_sc'] = ( data_test['Measured Values']- z_meas_min)/(z_meas_max -z_meas_min)

X_test = data_test[colum_sc]
y_test = data_test['wse_sim_sc']

#-------------------Data Train
 
X_train = pd.DataFrame()
for i in colum_sc:
    X_train[i] = data_tr['wse_sim_sc'].values
y_train =  data_tr['wse_sim_sc']
  

#-------------------Define Sequential model with 3 layers
model = keras.Sequential()
model.add(layers.Dense(100, activation="relu", input_shape=(reg_num,)))

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
EPOCHS = 100

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


wse_ann_tr_sc = model.predict(X_train)
wse_ann_tr = (wse_ann_tr_sc) * (z_max - z_min) + z_min

#---------------print some results
dist = pd.read_excel('dist.xlsx')

wse1 = wse_ann[0]
wse2 = wse_ann[38]
wse2_sim = data_test.iloc[38:76, 1]
wse3 = wse_ann[76]
wse3_sim = data_test.iloc[76:114, 1]
wse4 = wse_ann[114]
wse4_sim = data_test.iloc[114:152, 1]
wse1_sim = data_test.iloc[0:38, 0]

plt.figure()
axes = plt.gca()
axes.set_ylim([446,450])
plt.scatter(dist, wse1, facecolor='blue', label='ANN-Test1')
plt.scatter(dist, wse1_sim, facecolor='red', label='iRIC')
plt.legend()

plt.figure()
axes = plt.gca()
axes.set_ylim([446,450])
plt.scatter(dist, wse2, facecolor='blue', label='ANN-Trest2')
plt.scatter( dist, wse2_sim, facecolor='red', label='iRIC')
plt.legend()

plt.figure()
axes = plt.gca()
axes.set_ylim([446,450])
plt.scatter(dist, wse3, facecolor='blue', label='ANN-Test3')
plt.scatter( dist, wse3_sim, facecolor='red', label='iRIC')
plt.legend()

plt.figure()
axes = plt.gca()
axes.set_ylim([446,450])
plt.scatter(dist, wse4, facecolor='blue', label='ANN-Test4')
plt.scatter( dist, wse4_sim, facecolor='red', label='iRIC')
plt.legend()

plt.figure()
axes = plt.gca()
axes.set_ylim([446,450])
plt.scatter(dist, wse_ann_tr[29], facecolor='blue', label='ANN-training set')
plt.scatter( dist, data_tr['Calculatioin Result'].iloc[1102:1140], facecolor='red', label='iRIC')
plt.legend()

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
# plt.savefig('Final_Performance1', dpi = 300)
# # plt.savefig('n={}'.format(X_train.shape))


# #-------------save the model, data, and results

# with open('trainHistoryDict', 'wb') as file_pi:
#         pickle.dump(results.history, file_pi)
        
# model.save('GA_ANN')# 

# # save data
# pickle_out = open('X_train_final', 'wb')
# pickle.dump(X_train, pickle_out)
# pickle_out.close()

# pickle_out = open('y_train_final', 'wb')
# pickle.dump(y_train, pickle_out)
# pickle_out.close()

# pickle_out = open('X_test', 'wb')
# pickle.dump(X_test, pickle_out)
# pickle_out.close()

# pickle_out = open('y_test', 'wb')
# pickle.dump(y_test, pickle_out)
# pickle_out.close()

# pickle_out = open('n_max', 'wb')
# pickle.dump(n_max, pickle_out)
# pickle_out.close()

# pickle_out = open('n_min', 'wb')
# pickle.dump(n_min, pickle_out)
# pickle_out.close()


# pickle_out = open('z_max', 'wb')
# pickle.dump(z_max, pickle_out)
# pickle_out.close()

# pickle_out = open('z_min', 'wb')
# pickle.dump(z_min, pickle_out)
# pickle_out.close()

# pickle_out = open('z_meas_max', 'wb')
# pickle.dump(z_meas_max, pickle_out)
# pickle_out.close()

# pickle_out = open('z_meas_min', 'wb')
# pickle.dump(z_meas_min, pickle_out)
# pickle_out.close()

# pickle_out = open('data_tr', 'wb')
# pickle.dump(data_tr, pickle_out)
# pickle_out.close()








