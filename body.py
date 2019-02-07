# =============================================================================
# # -*- coding: utf-8 -*-
# """
# Created on Thu Sep 27 00:11:18 2018
# 
# @author: Nadim
# """
# 
# min_all=np.min(dt1_b)
# print(min_all)
# max_all=np.max(dt1_b)
# dt1_b_norm=(dt1_b-min_all)/(max_all-min_all)
# dt1_b_norm=dt1_b_norm.values
# dt_norm_1=np.transpose(dt1_b_norm)
# ### converted dataframe to np matrix
# np.shape(dt_norm_1)
# 
# ##### Answer to 1(b)_i_ii
# dt1_cov=np.cov(dt_norm_1)
# print(dt1_cov)
# np.shape(dt1_cov)
# #####returns the largest eigen values and the correspoidng vectos
# i=5
# dt1_eig_value,dt1_eig_vector=scipy.sparse.linalg.eigsh(dt1_cov,k=i)
# e_val=dt1_eig_value
# e_vec=dt1_eig_vector
# ### returns all the vlaues and vectors
# e_v,e_vector=np.linalg.eig(dt1_cov)
# e_v=np.sort(e_v)
# #####Largest 5 eigen values
# e_val
# ####
# ####Answer to 1((b)_iii)
# ##### plotting eigen values in ranges
# plt.hist(e_v)
# e_v[e_v>=0]
# np.shape(e_v)
# ###### would imaginary numbers be good choice for eigen values in PCA
# #######
# ##############
# ###Answer to 1(b)_iv
# ### making the feature vector
# feature_vector=np.transpose(dt1_eig_vector)
# rowData=np.transpose(dt1_b_norm)
# np.shape(feature_vector)
# np.shape(rowData)
# new_Data=feature_vector.dot(rowData)
#                        
# 
# =============================================================================
