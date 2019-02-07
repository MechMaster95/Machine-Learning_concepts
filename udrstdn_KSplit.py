# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 11:38:48 2018

@author: Nadim
"""
import pandas as pd
import numpy as np
import scipy.sparse.linalg
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

import numpy as np
from sklearn.model_selection import KFold
dt1=pd.read_csv("hw2q5.csv")
dt1_b=dt1.loc[:,dt1.columns[1:]]
print(dt1_b)
kf = KFold(5)
print(kf)

for train_index, test_index in kf.split(dt1):
    features_train=[]
    class_test=[]
    features_test=[]
    class_train=[]
    
    for i in train_index:
        temp=[]
        for j in range(1,5):
            temp.append(dt1[i][j])
        features_train.append(temp[i,:])
        class_train.append(dt1[i][5])
            
    
    for i in test_index:
        temp=[]
        for j in range(1,5):
            temp.append(dt1[i][j])
        features_test.append(temp[i,:])
        class_test.append(dt1[i][5])
            
            