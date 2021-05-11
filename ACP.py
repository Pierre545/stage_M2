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
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn import preprocessing
from eofs.standard import Eof
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

## Application de l'ACP à la matrice / module scikit

#On importe la matrice sauvegardé precedemment
matrice = np.transpose(np.loadtxt('/home/pierre/Desktop/Stage_M2/Espace_Dev_IRD/images/matrice', dtype=float))

##Application de l'ACP
pca = PCA(n_components = 18, random_state = 2020)
pca.fit(matrice)

#Composantes principales
PCs  = pca.transform(matrice)

#Fonctions orthogonale empirique
EOFs = pca.components_

#Pourcentage de la variance expliqué par chaque composante principale
eigenvalues = pca.explained_variance_ratio_*100

#Somme cumulative des pourcentages
np.cumsum(pca.explained_variance_ratio_*100)

#%%

## Application de l'ACP à la matrice / module eofs

#On importe la matrice sauvegardé precedemment
matrice = np.transpose(np.loadtxt('/home/pierre/Desktop/Stage_M2/Espace_Dev_IRD/images/matrice', dtype=float))

#On créé une instance Eof
solver = Eof(matrice)

#Calcul des composantes principales
pcs_eofs = solver.pcs(npcs=18, pcscaling=0)

#Calcul des Empirical orthogonal functions (EOF)
EOFs_eofs = solver.eofs(neofs=18, eofscaling=0)

#Pourcentage de la variance expliqué par chaque composante principale
eigenvalue_eofs = solver.eigenvalues(neigs=18)


#%%

#Expression de la variance selon la composante principale
features = range(pca.n_components_)
plt.figure(figsize=(15, 5))
plt.bar(features, pca.explained_variance_ratio_*100)
plt.xlabel('Caractéristiques des Composantes Principales')
plt.ylabel('Expression de la variance en %')
plt.xticks(features)
plt.title("Importance des différentes composantes principales")
plt.show()

#Affichage sous forme de tableau des valeurs des composantes principales
principalcomponents = PCs[:,:8]
principalDf = pd.DataFrame(data = principalcomponents, columns = ['pc1', 'pc2','pc3','pc4','pc5','pc6','pc7','pc8'])
principalDf.head()

#Affichage de la composante temporelle (PC) des EOFS
plt.plot(PCs[:,0],"-*")

#%%

#Extraction d'une partie de la matrice et d'une partie de la matrice après application de l'ACP
b1 = matrice[:,0:20]
b2 = EOFs[:,0:20]

#Affichage des variables b1 et b2
fig = plt.figure(figsize=(10,50))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
im1 = ax1.imshow(b1)
im2 = ax2.imshow(b2)


#fig.text(0.5, 0.04, 'TEMPS', ha='center')
#fig.text(0.04, 0.5, 'ESPACE', va='center', rotation='vertical')

ax1.set_xlabel('Temps')
ax1.set_ylabel('Espace')

ax1.set_title('Matrice')
ax2.set_title('EOF')

divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im1, cax=cax, orientation='vertical')

divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im2, cax=cax, orientation='vertical')

#%%

#Affichage et sauvegarde des reconstructions des composantes principales

for i in range (7):
    a = EOFs[i,:]
    b = a.reshape((1250,2250))

    plt.figure()
    plt.title("PCA "+str(i+1))
    plt.imshow(b,cmap='gray')
    plt.colorbar()
    plt.savefig('/home/pierre/Desktop/Stage_M2/Espace_Dev_IRD/images/zone_etude_test/ACP/PCA'+str(i+1)+'.png')
    plt.show()


#%%

#Création d'une copie de la variable matrice
matrice_diff = np.copy(matrice)

#Soustraction des trois premiers EOFs à la matrice_bis
for i in range(3):
    for j in range (np.shape(matrice_diff)[0]):
        matrice_diff[j,:] = matrice_diff[j,:] -  EOFs[i,:]

#Application de l'ACP à la matrice sortante
pca = PCA( random_state = 2020)
pca.fit(matrice_diff)

#Composantes principales
PCs_diff  = pca.transform(matrice_diff)

#Fonctions orthogonale empirique
EOFs_diff = pca.components_

#Pourcentage de la variance expliqué par chaque composante principale
eigenvalues_diff = pca.explained_variance_ratio_*100

#Somme cumulative des pourcentages
np.cumsum(pca.explained_variance_ratio_*100)


# Affichage des résultats
a = np.zeros((1,2812500))
b = np.zeros((1,2812500))

for i in range (18):
    a = a + matrice[i,:]
    b = b + matrice_diff[i,:]

a = a / 18
b = b / 18

a.resize((1250,2250))
b.resize((1250,2250))

#Affichage de la différence entre la matrice d'origine moyenne, et l'image moyenne d'origine auquelle ont été soustrait les 3 premiers EOFs
plt.figure()
plt.title("Elements EOFs soustrait à la matrice d'origine")
plt.imshow(a-b,cmap='gray')
plt.show()

#Affichage du premier EOFs de la matrice diff et de la matrice d'origine
EOF_1 = np.zeros((1,2812500))
EOF_1_bis = np.zeros((1,2812500))

EOF_1 = EOF_1 + EOFs[0,:]
EOF_1_bis = EOF_1_bis + EOFs_diff[0,:]

EOF_1.resize((1250,2250))
EOF_1_bis.resize((1250,2250))

fig = plt.figure(figsize=(20,100))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
im1 = ax1.imshow(EOF_1,cmap='gray')
im2 = ax2.imshow(EOF_1_bis,cmap='gray')

ax1.set_title('EOF_1 de matrice')
ax2.set_title('EOF_1 de matrice_diff')

divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im1, cax=cax, orientation='vertical')

divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im2, cax=cax, orientation='vertical')
