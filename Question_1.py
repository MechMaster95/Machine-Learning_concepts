# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 13:50:55 2018

@author: Nadim"""
import pandas as pd
import numpy as np
import scipy.sparse.linalg
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
dt1=pd.read_csv("hw2q1_train.csv")
dt2=pd.read_csv("hw2q1_test.csv")
print(len(dt1))
print(len(dt2))
print(dt1.head())
####Answer to 1(a)
len(dt1["Class"])
no_of_Rocks=len(dt1.loc[dt1["Class"]=="M"])
print(no_of_Rocks)
no_of_Mines=len(dt1.loc[dt1["Class"]=="R"])
print(no_of_Mines)

no_of_Rocks_t=len(dt2.loc[dt1["Class"]=="M"])
print(no_of_Rocks_t)
no_of_Mines_t=len(dt2.loc[dt1["Class"]=="R"])
print(no_of_Mines_t)
####Answer to 1(b)
dt1_b=dt1.loc[:,dt1.columns[0:-1]]
dt1_y=dt1.loc[:,dt1.columns[-1]]
dt2_b=dt2.loc[:,dt2.columns[0:-1]]
dt2_y=dt2.loc[:,dt1.columns[-1]]
var1=np.var(dt1_b)
min1=np.min(dt1_b)
max1=np.max(dt1_b)
diff=max1-min1
asc=np.sort(diff)
print(asc)
print(diff)
print(com)
print(min1)
print(max1)
print(var1)
var1=np.sort(var1)

class PCA:
          def __init__(self,cut_matrix_train,cut_matrix_test,p):
              self.cut_matrix_train=cut_matrix_train
              self.cut_matrix_test=cut_matrix_test
              self.p=p
#              self.cut_matrix_test=cut_matrix_test

          
          def norm(cmtr):
              min_all=np.min(cmtr)
              max_all=np.max(cmtr)
              dt_norm1=(cmtr-min_all)/(max_all-min_all)
              return dt_norm1
          
              
          def PCA_call(self):
# =============================================================================
# =============================================================================
# #               min_all=np.min(self.cut_matrix_train)
# #               max_all=np.max(self.cut_matrix_train)
# #               dt_norm=(self.cut_matrix_train-min_all)/(max_all-min_all)
# =============================================================================
              dt_norm=PCA.norm(self.cut_matrix_train)
              dt_norm1=PCA.norm(self.cut_matrix_test)
#              dt_norm1=PCA.norm(self.cut_matrix_test)
              dt_cov=(np.cov(np.transpose(dt_norm)))
              e_v,e_vector=np.linalg.eig(dt_cov)
              np.shape(e_v)
              e_v=np.sort(e_v)
              plt.hist(e_v)            
              dt_eig_value,dt_eig_vector=scipy.sparse.linalg.eigsh(dt_cov,k=self.p)
              print(dt_eig_value)
              feature_vector=np.transpose(dt_eig_vector)
              new_Data=feature_vector.dot(np.transpose(dt_norm1))
              print(new_Data)
              print(dt_eig_value)
              return(new_Data)
val=[2,4,8,10,20,40,60]
for idx,pc in enumerate(val):
    pc=5
    PCA_1 = PCA(dt1_b,dt1_b,pc)
    data_out=np.transpose(PCA_1.PCA_call())    
    np.shape(data_out)
    
    #####Define the model K-NN
    classifier = KNeighborsClassifier(n_neighbors=3,p=2,metric='euclidean')  
    #### Fit Model
    classifier.fit(data_out,dt1_y)
    np.shape(data_out)
    ####### Fitting is done, now test
    PCA_1 = PCA(dt1_b,dt2_b,pc)
    data_out=np.transpose(PCA_1.PCA_call())    
    np.shape(data_out)
    ####Test Model
    ##### dimesnioanlity Reduction for the test set
    test_data=data_out
    y_pred=classifier.predict(test_data)
    y_actual=dt2_y
    y_actual=np.array(y_actual)
    print(y_pred)
    print(y_actual)
    np.shape(y_actual)
    np.shape(y_pred)
    ###Evaulate Model
    cm = confusion_matrix(y_actual,y_pred)
    print(cm)
    accuracy_score(y_actual,y_pred)
    ####### produce_excel_sheet
    #####Produce graph accuracy vs P    
    


                       
            
