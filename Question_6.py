# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 18:01:19 2018

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
dt1=pd.read_csv("hw2q6.csv")
dt1_b=dt1.loc[:,dt1.columns[0:-1]]

def norm(cmtr):
              min_all=np.min(cmtr)
              max_all=np.max(cmtr)
              dt_norm1=(cmtr-min_all)/(max_all-min_all)
              return dt_norm1
          
dt1_norm=norm(dt1_b)
print(dt1_norm)
classifier = KNeighborsClassifier(n_neighbors=3,p=2,metric='euclidean')  