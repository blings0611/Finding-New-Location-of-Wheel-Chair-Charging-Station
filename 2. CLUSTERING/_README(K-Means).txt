
본 분석은 Jupyter Notebook (anaconda3) 에서 진행되었으며, 실행 환경(Jupyter Notebook, Python)에 맞춰 선택할 수 있도록, 'ipynb', 'py' 2가지의 형태로 제공하고 있음. 


--- STEP1) Import libararies

본 분석에서는 데이터셋 로드, 데이터 스케일링, 주성분 분석(Principal Component Analysis), 엘보우 메서드(Elbow Method), K평균 군집화(K-Means Clustering), 결과 저장 순으로 진행함.

따라서 첫째로 데이터 처리 및 분석을 위한 pandas, 과학적 계산을 위한 numpy, 머신러닝 알고리즘(군집 분석 등)을 위한 scikit-learn과 관련 수치해석을 위한 scipy, 과학 계산용 그래프를 표현하기 위한 matplotlib 라이브러리를 순차적으로 호출함. ---

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


--- STEP2) Load csv data

입출력이 이루어지는 작업 공간을 확인하고, 분석 데이터셋을 불러오기 위해 현재 작업 공간 내 위치한 'DATASET.csv' 파일을 가져온 후, 자료의 형태를 확인함(출력 결과물 또한 동일한 작업 공간에 저장). ---

import os 
os.getcwd() # Get my working directory (current working directory)

df=pd.read_csv('DATASET.csv', index_col='ID')
df.head()


--- STEP3) Data scaling

효과적인 데이터 알고리즘 학습을 위해 데이터 스케일링을 실시함.
원 데이터의 형태를 고려하여 Scikit-Learn에서 제공되는 스케일러 중 최대/최소값이 1 ,0이 되도록 하는 MinMaxScaler를 사용하였음.  ---

# Scaler: StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler

scaled = MinMaxScaler().fit_transform(df)
features = ['DISALBED', 'SHOPPING MALL', 'TAXI', 'EMPLOYEE','WELFARE CENTER','CARD']
pd.DataFrame(scaled, columns=features)


--- STEP4) Principal Component Analysis

6개의 변수의 차원을 축소하고자 주성분 분석을 실시함.
우선적으로 주성분 수를 선택하기 위해, 설명된 분산의 비율이 80%가 넘는 지점인 3개의 차원으로 축소하였음.
(여기서 '80%'란 비율은 정해진 규칙은 아니나 기존의 연구자들의 경험에 의해 권장되는 수치로 본 분석에서 참고하였음.) ---

pca = PCA(n_components=3) # n=3, 0.835 (83.5%) (Explained variance ratio should be at least 0.8)
values_pca = pca.fit_transform(scaled)
principalDf = pd.DataFrame(data=values_pca, columns = ['PC-1', 'PC-2','PC-3'])
print('Explained variance ratio :', pca.explained_variance_ratio_)

principalDf


--- STEP5 Elbow method

K-Means 클러스터링 분석을 위해 데이터셋의 클러스트 수를 결정할 필요가 있다.
본 분석에서는 보편적으로 많이 사용되고 있는 엘보우 메서드를 사용하였고, 3개의 군집이 최적이라 도출하였음. ---

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

elbow_method(values_pca) # Optimal K for K-Means = 3


--- STEP6 

앞서 도출된 최적의 군집 수 3개를 적용하여, K-Means 클러스터링 분석을 하였음.
클러스터링 분석 결과값이 이상 없이 도출되었는지 3차원 그래프를 통해 확인 후 라벨링하여 작업공간 내 'BARRIER_FREE.xlsx' 파일로 출력함. ---


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


# Labeling

df['CLUSTERING RESULT']=kmeans.labels_
df.head()

df.to_excel('BARRIER_FREE.xlsx')
