# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 10:44:30 2018

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
dt1=pd.read_csv("hw2q5.csv")
print(dt1)
kf = KFold(n_splits=5)
n=kf.get_n_splits(dt1)
m=kf.split(dt1)
print(n)
print(m)
