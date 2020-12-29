#!/usr/bin/env python
# coding: utf-8

# # Wheelchair Charging Stations for the Disabled in Seoul

# ## Import libraries

# In[2]:


# For data manipulation and analysis
import pandas as pd
from pandas import Series, DataFrame

# For scientific computing
import numpy as np

# For clustering algorithm (K-Means)
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial import distance
from scipy.spatial.distance import cdist, pdist
from sklearn.metrics import euclidean_distances
from sklearn.metrics import silhouette_score

# For visulization
import matplotlib.pyplot as plt
import matplotlib
from math import sqrt
from mpl_toolkits.mplot3d import Axes3D


# ## Load csv data

# In[3]:


import os 
os.getcwd() # Get my working directory (current working directory)


# In[4]:


df=pd.read_csv('DATASET.csv', index_col='ID')
df.head()


# ## Data scaling (MinMaxScaler)

# In[5]:


# Scaler: StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler

scaled = MinMaxScaler().fit_transform(df)
features = ['DISALBED', 'SHOPPING MALL', 'TAXI', 'EMPLOYEE','WELFARE CENTER','CARD']
pd.DataFrame(scaled, columns=features)


# ## Principal Component Analysis

# In[6]:


pca = PCA(n_components=3) # n=3, 0.835 (83.5%) (Explained variance ratio should be at least 0.8)
values_pca = pca.fit_transform(scaled)
principalDf = pd.DataFrame(data=values_pca, columns = ['PC-1', 'PC-2','PC-3'])
print('Explained variance ratio :', pca.explained_variance_ratio_)


# In[7]:


principalDf


# ## Elbow method

# In[9]:


# Using the elbow method to determine the optimal number of clusters for k-means clustering

def elbow_method(data):
    K = range(2,6) 
    KM = [KMeans(n_clusters=k).fit(data) for k in K] 
    centroids = [k.cluster_centers_ for k in KM] 
    D_k = [cdist(data, cent, 'euclidean') for cent in centroids] 
    cIdx = [np.argmin(D,axis=1) for D in D_k]
    dist = [np.min(D,axis=1) for D in D_k]
    avgWithinSS = [sum(d)/data.shape[0] for d in dist]

    # Total with-in sum of square
    wcss = [sum(d**2) for d in dist]
    tss = sum(pdist(data)**2)/data.shape[0]
    bss = tss-wcss
    

    # Elbow curve
    fig = plt.figure() 
    ax = fig.add_subplot(111)
    
    kIdx = 1
    ax.plot(K, avgWithinSS, 'b*-')
    ax.plot(K[kIdx], avgWithinSS[kIdx], marker='o', markersize=15, 
    markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
    
    kIdx = 2
    ax.plot(K, avgWithinSS, 'b*-')
    ax.plot(K[kIdx], avgWithinSS[kIdx], marker='o', markersize=10, 
    markeredgewidth=2, markeredgecolor='b', markerfacecolor='None')
    
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Average within-cluster sum of squares')
    plt.title('Elbow for K-Means clustering')


# In[10]:


elbow_method(values_pca) # Optimal K for K-Means = 3


# ## K-Means Clustering

# In[13]:


# K-Means Clustering

fig = plt.figure(1, figsize=(9, 7)) 
ax = Axes3D(fig, elev=-150, azim=230)

kmeans = KMeans(init='k-means++', n_clusters=3, n_init=5) # Opitmal K = 3
kmeans.fit(values_pca)

ax.scatter(values_pca[:, 0], values_pca[:, 1], values_pca[:, 2], 
           c=kmeans.labels_.astype(np.float), s=300
          )

cntr = kmeans.cluster_centers_
ax.scatter(cntr[0][0],cntr[0][2],cntr[0][1],c = 'red',marker="x",s=1000)
ax.scatter(cntr[1][0],cntr[1][2],cntr[1][1],c = 'red',marker="x",s=1000)
ax.scatter(cntr[2][0],cntr[2][2],cntr[2][1],c = 'red',marker="x",s=1000)


# For visualization

ax.set_title('K-Means clustering (PCA-reduced data)\n'
             'Centroids are marked with red cross')

ax.set_xlabel("Principal Component 1")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("Principal Component 2")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("Principal Component 3")
ax.w_zaxis.set_ticklabels([])


# In[14]:


#Labeling

df['CLUSTERING RESULT']=kmeans.labels_


# In[15]:


df.head()


# In[16]:


df.to_excel('BARRIER_FREE.xlsx')

