#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 19:22:30 2021

@author: audisio
"""

dataset = gdal.Open('/media/audisio/Maxtor/S1_PierreAudisio/Data_2/DATA_VV/VV/20201223T092239_20201223T092304/Sigma0_VV.img')
        
                #Conversion du fichier en array et extraction d'une zone d'interet
band = dataset.GetRasterBand(1)
x = 10500 
y = 6000
array = band.ReadAsArray(1050,950,x,y)
Image.fromarray(array).save('/media/audisio/Maxtor/resultats/07_07_2021/test.tif')
