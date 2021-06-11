#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 10:16:19 2021

@author: pierre
"""
import numpy as np

data_set = np.load()
file = data_set['arra_0']
file_flat = file.flatten()
ax,ay = np.histogram(file, bins = np.linspace(-1,1,10000))
list_value = []
index = []

n = 0
for i in (ax):
    n = n+1
    if (i != 0):
        list_value.append(i)
        index.append(n)


bin_value = []
for i in (index):
    bin_value.append(ay[i])
    
test_values = np.linspace(0,255,11) 

for i in range (len(file_flat)):
    if (file_flat[i] >= bin_value[0] and file_flat[i] <= bin_value[1]):
        file_flat[i] = test_values[1]
        
    if (file_flat[i] >= bin_value[1] and file_flat[i] <= bin_value[2]):
        file_flat[i] = test_values[2]
        
    if (file_flat[i] >= bin_value[2] and file_flat[i] <= bin_value[3]):
        file_flat[i] = test_values[3]

    if (file_flat[i] >= bin_value[3] and file_flat[i] <= bin_value[4]):
        file_flat[i] = test_values[4]

    if (file_flat[i] >= bin_value[4] and file_flat[i] <= bin_value[5]):
        file_flat[i] = test_values[5]

    if (file_flat[i] >= bin_value[5] and file_flat[i] <= bin_value[6]):
        file_flat[i] = test_values[6]

    if (file_flat[i] >= bin_value[6] and file_flat[i] <= bin_value[7]):
        file_flat[i] = test_values[7]

    if (file_flat[i] >= bin_value[7] and file_flat[i] <= bin_value[8]):
        file_flat[i] = test_values[8]

    if (file_flat[i] >= bin_value[8] and file_flat[i] <= bin_value[9]):
        file_flat[i] = test_values[9]

    if (file_flat[i] >= bin_value[9] and file_flat[i] <= bin_value[10]):        
        
        file_flat[i] = test_values[10]

    else :
        file_flat[i] = 0