#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 10:16:19 2021

@author: pierre
"""
import numpy as np
from matplotlib import pyplot as plt
import os


def transfo_hist( x ):
    
    file_flat = x.flatten()
    #Evaluer avec et sans flattening
    
    #Les valeurs etudie sont toute comprise entre -1 et 1
    ax,ay = np.histogram(file, bins = np.linspace(-1,1,10000))
    
    
    bin_value = np.zeros(3)
    nb_value = 0
    
    n = 0
    for i in (ax):
        n = n+1
        if (i >= nb_value):
            nb_value = i
            
            bin_value[0] = ay[n-1]
            bin_value[1] = ay[n]
            bin_value[2] = ay[n+1]
            
 
    
    for j in range (len(file_flat)):
        if (file_flat[j] >= bin_value[1]):
            file_flat[j] = 200
        if ( file_flat[j] <= bin_value[1]):
            file_flat[j] = 100
        else :
            file_flat[j] = 0
        
        
    img = file_flat.reshape((6000,10500))
        
    return( img )



path_1 = '/media/pierre/Maxtor/resultats/08_06_2021/image_origine/EOF_image_origine'
files_1 = os.listdir(path_1)
for name_1 in files_1: 
    print(path_1+name_1) 
    data_set = np.load('/media/pierre/Maxtor/resultats/08_06_2021/image_origine/EOF_image_origine/EOF_5.npz')
    file = data_set['arr_0']
    b = transfo_hist( file )
    
    
    plt.figure()
    plt.title(name_1)
    plt.imshow(b)
    plt.show()