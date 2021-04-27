# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 09:52:52 2021

@author: Pierre Audisio
"""
import time 
import sys
import numpy as np
import scipy
import os
from pathlib import Path
from skimage import io
from osgeo import gdal
from matplotlib import pyplot as plt

#%%
startTime = time.time()
executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))
#%%

files = os.listdir('/media/pierre/Volume/T2/tif_image/')

dataset = gdal.Open('/home/pierre/Desktop/Stage_M2/Espace_Dev_IRD/Data/T2/VHdB_20170111T093921.tif')

for name in files:  
 
    print(name)
    dataset = gdal.Open('/media/pierre/Volume/T2/tif_image/'+name)

    print()
    print("MetaData",dataset.GetMetadata())
    print()
    print("Nombre de bandes",dataset.RasterCount)
    print("Hauteur:",dataset.RasterXSize)
    print("Largeur:",dataset.RasterYSize)

    Array1 = np.array(dataset.GetRasterBand(1).ReadAsArray())
    print("Cropping")
    crop = Array1[2500:3750,5000:7250]

    plt.figure()
    plt.imshow(crop)
    plt.show()
    
    print("Taille en bytes:",sys.getsizeof(crop))

