# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 20:53:18 2015

@author: user
"""

import arff,matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA

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
show2D(pcaReduction(dataset))