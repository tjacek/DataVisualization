# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 20:53:18 2015

@author: user
"""
import arff,sklearn.ensemble 
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn import manifold

def pcaReduction(X,dim=2):
    pca = PCA(n_components=dim)
    return pca.fit(X).transform(X),'PCA'

def mdaReduction(X,dim=2):
    clf = manifold.MDS(n_components=dim, n_init=1, max_iter=100)
    return clf.fit_transform(X),'MDA'

def lleReduction(X,dim=2,n_neighbors=5):
    clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=dim,
                                      method='standard')
    return clf.fit_transform(X),'LLE'

def spectralReduction(X,dim=2):
    embedder = manifold.SpectralEmbedding(n_components=dim, random_state=0,
                                     eigen_solver="arpack")
    return embedder.fit_transform(X),'Spectral'

def ensembleReduction(X,dim=2):
    hasher = sklearn.ensemble.RandomTreesEmbedding(n_estimators=200, random_state=0,
                                       max_depth=5)
    X_transformed = hasher.fit_transform(X)
    pca = TruncatedSVD(n_components=dim)
    X_reduced = pca.fit_transform(X_transformed)
    return X_reduced,'randomForest'

def tsneReduction(X,dim=2):
    tsne = manifold.TSNE(n_components=dim, init='pca', random_state=0)
    return tsne.fit_transform(X),'TSNE'

def reduceDataset(name,newDim=50,reduction=pcaReduction):
    dataset=arff.readArffDataset(name)  
    reducType=dataset.applyReduction(newDim,reduction)
    arffStr=str(dataset)
    output=name.replace(".arff","_"+reducType+str(newDim)+".arff")
    arffFile = open(output, 'w')
    arffFile.write(arffStr)
    arffFile.close()