#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 08:36:13 2022

@author: vech
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from skimage import morphology #for measuring things in the masks


import groupXY_functions as util #custom-made functions with e.g. kNN classifier, you can also use sklearn
from sklearn.neighbors import KNeighborsClassifier

import pickle #for saving/loading trained classifiers





#The function that should classify new images. The image and mask are the same size, and are already loaded using plt.imread
def classify(img, mask):
    
    
     #Do any kind of processing here that you also did to the training images (like resizing)
         
    
     #Extract features (and scale them if you did that for training)
     a, p = util.measure_area_perimeter(mask)
     x = [[a,p]]
     
     
     #Load the trained classifier
     classifier = pickle.load(open('groupXY_classifier.sav', 'rb'))
    
    
     #Use it on this example to predict the label 
     pred_label = classifier.predict(x)
     print('predicted label is ', pred_label)
     return pred_label
 
    
 #-------This part is just for testing the function in clas
file_data = 'data/example_ground_truth.csv'
path_image = 'data/example_image'
path_mask = 'data/example_segmentation'
file_features = 'features/features.csv'

df = pd.read_csv(file_data)

#We just test on one image
image_id = list(df['image_id'])
i = 0

file_image = path_image + os.sep + image_id[i] + '.jpg'
file_mask = path_mask + os.sep + image_id[i] + '_segmentation.png'
img = plt.imread(file_image)
mask = plt.imread(file_mask)


classify(img,mask)

#-------End of part for testing 
