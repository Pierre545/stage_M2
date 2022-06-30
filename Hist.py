#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 17:41:24 2022

@author: pierreaudisio
"""

import time 
import numpy as np


def hist( x, *bins ):
    
    startTime = time.time()

    r,c = np.shape(x)
    x = x.flatten()
    hist = np.zeros(len(x))
    
    bins = np.arange(0,len(x)+1,1)
    
    if (len(bins)!=(len(x)+1)):
        print("Les intervalles ne peuvent pas etre accepté")
        
    
    for i in (x):
        n = 0
        for j in range (len(bins)-1):
            if(bins[j]<=i<bins[j+1]):
                hist[n] = hist[n] + 1 
            n = n+1
                
                
    print(hist,bins)
    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))
    
    return(hist)
