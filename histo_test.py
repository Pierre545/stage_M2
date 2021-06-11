#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 10:16:19 2021

@author: pierre
"""
import numpy as np
from matplotlib import pyplot as plt


data_set = np.load('/media/audisio/Maxtor/resultats/08_06_2021/image_origine/EOF_image_origine/EOF_20.npz')
file = data_set['arr_0']
file_flat = file.flatten()
ax,ay = np.histogram(file, bins = np.linspace(-1,1,10000))

list_value = []
index = []
bin_value = []

n = 0
for i in (ax):
    n = n+1
    if (i != 0):
        list_value.append(i)
        index.append(n)

for i in (index):
    bin_value.append(ay[i])
    


#%%

def transfo_hist( x ):
    
    #file_flat = x.flatten()
    #Evaluer avec et sans flattening
    
    #Les valeurs etudie sont toute comprise entre -1 et 1
    ax,ay = np.histogram(file, bins = np.linspace(-1,1,10000))
    
    
    bin_value = np.zeros(3)
    nb_value = 0
    
    
    for i in (ax):
        if (i >= nb_value[1]):
            nb_value = i
            
            bin_value[0] = ay[i-1]
            bin_value[1] = ay[i]
            bin_value[2] = ay[i+1]
            
 
    
    for i in range (len(file_flat)):
        if (file_flat[i] >= bin_value[2]):
            file_flat[i] = 150
        if ( file_flat[i] <= bin_value[1]):
            file_flat[i] = 150
        else :
            file_flat[i] = 0
        
        
    img = file_flat.reshape((6000,10500))
        
    return( img )



plt.imshow(b)