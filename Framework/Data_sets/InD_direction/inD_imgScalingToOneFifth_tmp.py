#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 15:45:03 2023

@author: anna
"""
import cv2


for i in range(33):
    if i < 10:
        img = cv2.imread('./inD_TrainingData/inD-dataset-v1.0/data/0'+str(i)+'_background.png', 0)
    else:
        img = cv2.imread('./inD_TrainingData/inD-dataset-v1.0/data/'+str(i)+'_background.png', 0)
    
    scale_percent = 1/5 # percent of original size
    width = int(img.shape[1] * scale_percent)
    height = int(img.shape[0] * scale_percent)
    dim = (width, height)
      
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
     
 
