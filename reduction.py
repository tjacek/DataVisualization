# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 20:53:18 2015

@author: user
"""

import arff,matplotlib.pyplot as plt,random
from sklearn.decomposition import PCA
import matplotlib.cm as cmx,numpy as np
import matplotlib.colors as colors
from sklearn import manifold
from mpl_toolkits.mplot3d import Axes3D

def pcaReduction(X,dim=2):
    pca = PCA(n_components=dim)
    return pca.fit(X).transform(X)

def mdaReduction(X,dim=2):
    clf = manifold.MDS(n_components=dim, n_init=1, max_iter=100)
    return clf.fit_transform(X)

def lleReduction(X,dim=2,n_neighbors=20):
    clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=dim,
                                      method='standard')
    return clf.fit_transform(X)

def spectralReduction(X,dim=2):
    embedder = manifold.SpectralEmbedding(n_components=dim, random_state=0,
                                     eigen_solver="arpack")
    return embedder.fit_transform(X)

def tsneReduction(X,dim=2):
    tsne = manifold.TSNE(n_components=dim, init='pca', random_state=0)
    return tsne.fit_transform(X)

def showReduction(bunch,reduction,name='dataset'):
    X = bunch.data
    y = bunch.target
    target_names = bunch.target_names

    X_r=reduction(X,2)
    N=len(target_names)
    
    numbers=range(N)
  
    plt.figure()
    pointShape='ovs'
    for i, target_name in zip( numbers, target_names):
        #plt.text(X_r[y == i, 0], X_r[y == i, 1],str(target_name),
        #         color=plt.cm.Set1(y[i] / 10.),
        #         fontdict={'weight': 'bold', 'size': 9})
        m=pointShape[i / 9]
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1],c=getColor(i),marker=m, label=target_name)
    plt.legend()
    plt.title('PCA of '+name)
    plt.show()
    
def showReduction3D(bunch,reduction,name='dataset'):
    X = bunch.data
    y = bunch.target
    target_names = bunch.target_names

    X_r=reduction(X,3)
    N=len(target_names)
    numbers=range(N)
  
    fig=plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    pointShape='ovs'
    for i, target_name in zip( numbers, target_names):
        m=pointShape[i / 9]
        color=getColor(i)
        ax.scatter(X_r[y == i, 0], X_r[y == i, 1],X_r[y == i,2],c=color,marker=m, label=target_name)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.title('PCA of '+name)
    plt.show()
    

def getArea(index):
    div=index / 9
    return np.pi * (3+div)**2
    
def getColor(index):
    cls="bgrcmykw"
    i=index % len(cls)
    return cls[i]
    #if index<len(colors):
    #    return cls[index]
    #return np.random.rand(3,1)

dataset=arff.readArffDataset("C:/Users/user/Desktop/kwolek/output/3_8_4_0.arff")
showReduction3D(dataset.toBunch(),tsneReduction)
#show2D(pcaReduction(dataset))