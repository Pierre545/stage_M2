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
import matplotlib.image as mpimg
from sklearn.decomposition import PCA
from pathlib import Path
from skimage import io
from osgeo import gdal
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

#%%
#Permet de mesurer le temps d'execution
startTime = time.time()
executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))
#%%

#Extraction des fichiers .tif, et croppage d'une zone d'étude

files = os.listdir('/media/pierre/Volume/T2/tif_image/')
for name in files:  
 
    print(name)
    dataset = gdal.Open('/media/pierre/Volume/T2/tif_image/'+name)

    #Impression des données relatives au fichier .tif
    print()
    print("MetaData",dataset.GetMetadata())
    print()
    print("Nombre de bandes",dataset.RasterCount)
    print("Hauteur:",dataset.RasterXSize)
    print("Largeur:",dataset.RasterYSize)

    #Conversion fichier .tif à array
    Array1 = np.array(dataset.GetRasterBand(1).ReadAsArray())
    
    #Extraction d'une zone d'interet
    crop = Array1[2500:3750,5000:7250]
  
    np.savetxt('/home/pierre/Desktop/Stage_M2/Espace_Dev_IRD/images/crop/'+name+'_crop', crop, fmt='%d')
    

    #Affichage
    plt.figure()
    plt.imshow(crop)
    plt.show()
    
    #Impression de la taille de la zone d'études en bytes
    print("Taille en bytes:",sys.getsizeof(crop))
    
size = np.size(crop)

#%%

#Création d'une matrice contenant les différents crop de la zone d'étude

files = os.listdir('/home/pierre/Desktop/Stage_M2/Espace_Dev_IRD/images/crop/')

matrice = np.zeros([size,len(files)])
n = 0

for name in files:  
 
    print(name) 
    data = np.loadtxt('/home/pierre/Desktop/Stage_M2/Espace_Dev_IRD/images/crop/'+name, dtype=float)
    data = data.flatten()
    matrice[:,n] = data
    
    n = n+1
    
#%%
    
# Application de l'ACP à la matrice
    
pca = PCA()
matrice_PCA  = pca.fit_transform(matrice)

#%%

#Extraction d'une partie de la matrice et d'une partie de la matrice après application de l'ACP
b1 = matrice[0:200,0:]
b2 = matrice_PCA[0:200,0:]

#Affichage des variables b1 et b2
fig = plt.figure(figsize=(10,30))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
im1 = ax1.imshow(b1)
im2 = ax2.imshow(b2)


#fig.text(0.5, 0.04, 'TEMPS', ha='center')
#fig.text(0.04, 0.5, 'ESPACE', va='center', rotation='vertical')

ax1.set_xlabel('Temps')
ax1.set_ylabel('Espace')

ax2.set_xlabel('Temps')
ax2.set_ylabel('Espace')

ax1.set_title('Matrice')
ax2.set_title('Matrice_ACP')

divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im1, cax=cax, orientation='vertical')

divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im2, cax=cax, orientation='vertical')

