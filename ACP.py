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
from sklearn.preprocessing import StandardScaler
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
  
    #Sauvetage des différentes extractions dans un fichier exterieur (permet de les réimporter facilement)
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

#Lecture du fichier contenant les variables en format .txt des zones d'intéret 
files = os.listdir('/home/pierre/Desktop/Stage_M2/Espace_Dev_IRD/images/crop/')

#Création d'un matrice remplis de 0
matrice = np.zeros([size,len(files)])
n = 0

#Remplissage de la matrice
for name in files:  
 
    print(name) 
    data = np.loadtxt('/home/pierre/Desktop/Stage_M2/Espace_Dev_IRD/images/crop/'+name, dtype=float)
    data = data.flatten()
    matrice[:,n] = data
    
    n = n+1
    
#Sauvetage de la matrice
np.savetxt('/home/pierre/Desktop/Stage_M2/Espace_Dev_IRD/images/matrice', matrice, fmt='%d')
    
#%%
    
# Pretraitement et application de l'ACP à la matrice

#On importe la matrice sauvegardé precedemment
matrice = np.loadtxt('/home/pierre/Desktop/Stage_M2/Espace_Dev_IRD/images/matrice', dtype=float)


##Mettre les données sous le meme orde de grandeur

scale = StandardScaler()

#Calcul de la moyenne et de l'écart-type
scale.fit(matrice)

#Transformation des données
matrice_scale = scale.transform(matrice)


##Application de l'ACP

pca = PCA(n_components = 18, random_state = 2020)
pca.fit(matrice_scale)

#Composantes principales
matrice_PC  = pca.transform(matrice_scale)

#Fonctions orthogonale empirique
EOFs = pca.components_

#Pourcentage de la variance expliqué par chaque composante principale
pca.explained_variance_ratio_*100

#Somme cumulative des pourcentages
np.cumsum(pca.explained_variance_ratio_*100)

#%%

#Extraction d'une partie de la matrice et d'une partie de la matrice après application de l'ACP
b1 = matrice[0:200,0:]
b2 = matrice_PC[0:200,0:]

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

ax1.set_title('Matrice')
ax2.set_title('Matrice_ACP')

divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im1, cax=cax, orientation='vertical')

divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im2, cax=cax, orientation='vertical')

#%%

#Affichage et sauvegarde des reconstructions des composantes principales

for i in range (7):
    a = matrice_PCA[:,i]
    b = a.reshape((1250,2250))

    plt.figure()
    plt.title("PCA "+str(i+1))
    plt.imshow(b,cmap='gray')
    plt.colorbar()
    plt.savefig('/home/pierre/Desktop/Stage_M2/Espace_Dev_IRD/images/zone_etude_test/ACP/PCA'+str(i+1)+'.png')
    plt.show()

#%%
    
#Création de deux matrices, l'une contenant une image moyenne de la série temporelle de la zone d'étude, l'autre contenant la somme des ACP    
a = np.zeros((1,2812500))
b = np.zeros((1,2812500))

for i in range (18):
    a = a + matrice[:,i]
for i in range (3):
    b = b + matrice_PC[:,i]

a = a / 18

a.resize((1250,2250))
b.resize((1250,2250))


#Affichage des deux matrices et de leurs différences
fig = plt.figure(figsize=(20,100))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)
im1 = ax1.imshow(a,cmap='gray')
im2 = ax2.imshow(b,cmap='gray')
im3 = ax3.imshow(a-b,cmap='gray')

ax1.set_title('Image moyenne de la zone étudiée')
ax2.set_title('Somme des ACP 1,2 et 3')
ax3.set_title('Image moyenne - Somme des ACP 1,2 et 3')

divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im1, cax=cax, orientation='vertical')

divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im2, cax=cax, orientation='vertical')

divider = make_axes_locatable(ax3)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im3, cax=cax, orientation='vertical')



