# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 20:53:18 2015

@author: user
"""

import arff,matplotlib.pyplot as plt,numpy as np
from sklearn.decomposition import PCA

def showPCA(bunch,name='dataset'):
    X = bunch.data
    y = bunch.target
    target_names = bunch.target_names

    pca = PCA(n_components=2)
    X_r = pca.fit(X).transform(X)
    N=len(target_names)
    #colors = list(np.random.rand(N))
    colors="bgrcmykwbgrcmykwbgrc"
    numbers=range(N)
  
    plt.figure()
    for c, i, target_name in zip(colors, numbers, target_names):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, label=target_name)
    plt.legend()
    plt.title('PCA of '+name)
    plt.show()

def pcaReduction(dataset):
    X=dataset.toMatrix()
    pca = PCA(n_components=2)
    X_r = pca.fit(X).transform(X)
    return X_r
    
def show2D(X_r):
    x=X_r[:,0]
    y=X_r[:,1]
    plt.scatter(x, y)
    plt.title('PCA')

dataset=arff.readArffDataset("C:/Users/user/Desktop/kwolek/output/3_8_4_0.arff")
showPCA(dataset.toBunch())
#show2D(pcaReduction(dataset))