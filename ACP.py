# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 09:52:52 2021

@author: Pierre Audisio
"""
import time 
import numpy as np
import os
import pandas as pd
from scipy import ndimage
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn import preprocessing
from eofs.standard import Eof
from osgeo import gdal
from matplotlib import pyplot as plt
#from mpl_toolkits.axes_grid1 import make_axes_locatableatable
from PIL import Image
from Lee_filter import *
from Average_filter import *

#%%
#Permet de mesurer le temps d'execution
startTime = time.time()
executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))
#%%
#Création d'une fonction Cumulative count cuts effectuant modifiant l'histogramme de l'image étudié (on retrouve la meme fonction sur QGIS)
def scaleCCC (x):
    return((x-np.nanpercentile(x,2))/(np.nanpercentile(x,98)-np.nanpercentile(x,2)))


#%%
def histogram_etirage( x ):

    x = x.flatten()
    ax,ay = np.histogram(x, bins = np.linspace(0,1,10000))
    
    cum_hist = np.cumsum(ay)
    cum_hist = (cum_hist * 255) / cum_hist[-1]

    image = np.interp(x, np.linspace(0, 1, 10000), np.round(cum_hist))
    image = image.reshape((6000,10500))
    
    #plt.hist(cum_hist,bins = np.arange(0,256,1))
    
    return(image)


#%%

#Extraction des fichiers .tif, et croppage d'une zone d'étude

path_1 = '/media/pierre/0ae37471-fa97-457a-bd2b-c1af82bba950/TC/test/'
files_1 = os.listdir(path_1)

for name_1 in files_1: 
    print(name_1)
    path_2 = path_1+name_1
    files_2 = os.listdir(path_2)
    
    for name in files_2: 
        if ('img' in name):
            if ('VV' in name):
                print()
                dataset = gdal.Open(path_2+'/'+name)
        
                #Conversion du fichier en array et extraction d'une zone d'interet
                band = dataset.GetRasterBand(1)
                x = 1000
                y = 260
                array = band.ReadAsArray(9500,17000,x,y)                          
                #a = histogram_etirage (array)
                #array = scaleCCC (array)
                #array = lee_filter(array, 7)
                #array = convolution2d(array,mask,0)
                #Sauvetage des différentes extractions dans un fichier exterieur (permet de les réimporter facilement)
                np.savez('/home/pierre/Desktop/Stage_M2/Espace_Dev_IRD/zone vegetation/array_VV/'+name_1+"_VV",array)
                #Image.fromarray(array).save('/home/pierre/Desktop/test/'+name_1+'.tif')

    
path_1 = '/media/pierre/0ae37471-fa97-457a-bd2b-c1af82bba950/TC/test/'
files_1 = os.listdir(path_1)

for name_1 in files_1: 
    path_2 = path_1+name_1
    files_2 = os.listdir(path_2)
    
    for name in files_2: 
        if ('img' in name):
            if ('VH' in name):
                print(name_1)
                dataset = gdal.Open(path_2+'/'+name)
        
                #Conversion du fichier en array et extraction d'une zone d'interet
                band = dataset.GetRasterBand(1)
                x = 1000
                y = 260
                array = band.ReadAsArray(9500,17000,x,y)
                #Sauvetage des différentes extractions dans un fichier exterieur (permet de les réimporter facilement)
                np.savez('/home/pierre/Desktop/Stage_M2/Espace_Dev_IRD/zone vegetation/array_VH/'+name_1+"_VH",array)
                


#%%
path = '/media/pierre/Maxtor/resultats/07_07_2021/Zone_seche/VH_VV_seche/VH_VV_seche_rapport'
files = os.listdir(path)
list_name =[]
for name in files:  
    list_name.append(name)
list_name = sorted(list_name)

n_1 = 0
n_2 = 1

for name in list_name:  
    if (n_2<56):
        print(list_name[n_1])
        print(list_name[n_2])
        print(n_2)
        
        data_1 = np.load(path + files[n_1])
        data_1 = data_1['arr_0']
        
        
        data_2 = np.load(path + files[n_2])
        data_2 = data_2['arr_0']
        
        array = data_2/data_1
        
        np.savez('/media/pierre/Maxtor/resultats/07_07_2021/Zone_seche/VH_VV_seche/VH_VV_seche_rapport'+list_name[n_2],array)
        
    n_1= n_1+2
    n_2= n_2+2
        
    
    
#%%
startTime = time.time()

#Création d'une matrice contenant les différents crop de la zone d'étude

#Lecture du fichier contenant les variables en format .npz des zones d'intéret 
path = '/media/pierre/Maxtor/resultats/23_08_2020/array/'
files = os.listdir(path)

list_name =[]
for name in files:  
    list_name.append(name)
list_name = sorted(list_name)

#Création d'un matrice remplis de 0
size = 54657934
matrice = np.zeros([size,len(files)])
n = 0

#Remplissage de la matrice
for name in list_name:  

    print(path+name) 
    data = np.load(path + name)
    data = data['arr_0']
    matrice[:,n] = data.flatten()
    n = n+1
    
#Sauvetage de la matrice
#np.savez('/media/pierre/0ae37471-fa97-457a-bd2b-c1af82bba950/matrice_VV_VH', matrice)

executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))

#%%

## Application de l'ACP à la matrice / module scikit

#On importe la matrice sauvegardé precedemment
# matrice = np.load('/media/pierre/Maxtor/resultats/18_06_2021/Lee_filter_3/matrice_VV_Leefilter_3.npz')
# matrice = matrice['arr_0']
matrice = np.transpose(matrice)

startTime = time.time()

##Application de l'ACP
#n_components = 18, random_state = 2020
pca = PCA()
pca.fit(matrice)

#Composantes principales
PCs  = pca.transform(matrice)

#Fonctions orthogonale empirique
EOFs = pca.components_

#Pourcentage de la variance expliqué par chaque composante principale
eigenvalues = pca.explained_variance_ratio_*100

#Somme cumulative des pourcentages
np.cumsum(pca.explained_variance_ratio_*100)

executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))

#%%

## Application de l'ACP à la matrice / module eofs

#On importe la matrice sauvegardé precedemment
#matrice = np.loadtxt('/home/audisio/Stage/matrice_VV', dtype=float)
matrice = np.transpose(matrice)

startTime = time.time()

#On créé une instance Eof
solver = Eof(matrice)

#Calcul des composantes principales
pcs_eofs = solver.pcs( pcscaling=0)

#Calcul des Empirical orthogonal functions (EOF)
EOFs_eofs = solver.eofs( eofscaling=0)

#Pourcentage de la variance expliqué par chaque composante principale
eigenvalue_eofs = solver.eigenvalues()

executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))

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
principalcomponents = pcs_eofs[:,:28]
principalDf = pd.DataFrame(data = principalcomponents, columns = ['pc1', 'pc2','pc3','pc4','pc5','pc6','pc7','pc8','pc9', 'pc10','pc11','pc12','pc13','pc14','pc15','pc16','pc17','pc18','pc19','pc20','pc21','pc22','pc23','pc24','pc25','pc26','pc27','pc28'])
                           #,index=['2017/02/06', '2017/03/02', '2017/03/14', '2017/03/26', '2017/04/19', '2017/05/01', '2017/05/13', '2017/05/25', '2017/06/06', '2017/06/18', '2017/06/30', '2017/07/12', '2017/07/24', '2017/08/05', '2017/08/17', '2017/08/29', '2017/09/10', '2017/09/22', '2017/10/04', '2017/10/16', '2017/10/28', '2017/12/27'])
principalDf.head()
principalDf.to_csv('/media/pierre/Maxtor/resultats/15_07_2021/PCA_EOF/VH/vegetation_inondee/VH.csv', index=False)

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

for i in range (28):
    print(i)
    a = EOFs_eofs [i,:]
    b = a.reshape((305,1200))
    Image.fromarray(b).save('/media/pierre/Maxtor/resultats/15_07_2021/PCA_EOF/VH/vegetation_inondee/VH_EOF/EOF_'+str(i)+'.tif')
    #b =histogram_etirage( a )
    
    # plt.figure()
    # plt.title("EOF"+str(i+1))
    # plt.imshow(b,cmap="gray")
    # plt.colorbar()
    # #plt.imsave('/media/audisio/Maxtor/resultats/08_06_2021/image_crop_EOF/EOF_'+str(i+1),b,format='tiff')
    # plt.show()


#%%

#Création d'une copie de la variable matrice
matrice_diff = np.copy(matrice)

#Soustraction des trois premiers EOFs à la matrice_bis
for i in range(3):
    for j in range (np.shape(matrice_diff)[0]):
        matrice_diff[j,:] = matrice_diff[j,:] -  EOFs[i,:]

#Application de l'ACP à la matrice sortante
pca = PCA( )
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
a = np.zeros((1,63000000))
b = np.zeros((1,63000000))

for i in range (19):
    a = a + matrice[i,:]
    b = b + matrice_diff[i,:]

a = a / 19
b = b / 19

a.resize((6000,10500))
b.resize((6000,10500))

#Affichage de la différence entre la matrice d'origine moyenne, et l'image moyenne d'origine auquelle ont été soustrait les 3 premiers EOFs
plt.figure()
plt.title("Elements EOFs soustrait à la matrice d'origine")
plt.imshow(a-b,cmap='gray')
plt.show()

#Affichage du premier EOFs de la matrice diff et de la matrice d'origine
EOF_1 = np.zeros((1,63000000))
EOF_1_bis = np.zeros((1,63000000))

EOF_1 = EOF_1 + EOFs[0,:]
EOF_1_bis = EOF_1_bis + EOFs_diff[0,:]

EOF_1.resize((6000,10500))
EOF_1_bis.resize((6000,10500))

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

#%%

PC1 = EOFs [0,:].flatten()
PC2 = EOFs [1,:].flatten()

plt.figure()
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(axis="both")
plt.plot(PC1,PC2,"*")
plt.show()
