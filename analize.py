# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 15:23:45 2015

@author: TP
"""

import arff
import numpy as np
import reduction
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from numpy import linalg as LA
from sklearn.lda import LDA
 
ccodes =['EU','BE','BG','CZ','DK','DE','EE',
         'IE','EL','ES','FR','IT','CY','LV',
         'LT','LU','HU','MT','NL','AT','PL',
         'PT','RO','SI','SK','FI','SE','UK',
         'HR','TR','IS','NO','CH','RS','MK']

crawl=['Hu',
'Ce','Dg','Ds','Dr','DD','DE','HE',
'Fe','Fo','Gr','Gh','Ha','Ko','Mf','Mi',
'Mu','Na','Op','Og','HO','Sp','Te','Tr',
'Vp','VS',]

colors=['b','g','r','y']

def showDataset():
    dataset=arff.readArffDataset("innovation_.arff")     
    #x,y=apply_LDA(dataset.data,dataset.target)
    x,y=apply_MDA(dataset.data)
    countryVisual(x,y,dataset.target)
    #countryVisual3D(x,y,z,dataset.target)

def showLDA():
    dataset=arff.readArffDataset("innovation_.arff")
    x,y,z=apply_LDA(dataset.data,dataset.target)
    countryVisual3D(x,y,z,dataset.target)
    
def countryVisual(x,y,cat):
    fig, ax = plt.subplots()
    c_p=[colors[i] for i in cat ]
    ax.scatter(x, y,c=c_p)
    for i,txt in enumerate(ccodes):
        ax.annotate(txt, (x[i],y[i]))

def countryVisual3D(x,y,z,cat):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    c_p=[colors[i] for i in cat ]
    ax.scatter(x,y,z,c=c_p)
    for i,txt in enumerate(ccodes):
        ax.annotate(txt, (x[i],y[i],z[i]))

def apply_PCA(X):
    pca = PCA(n_components=3)
    x_t=pca.fit(X).transform(X)
    cor1=[x_t[i][0] for i in xrange(len(x_t)) ]
    cor2=[x_t[i][1] for i in xrange(len(x_t)) ]
    print(pca.components_)
    print(pca.explained_variance_ratio_) 
    return cor1,cor2

def apply_LDA(X,Y):
    clf = LDA(n_components=2)
    clf.fit(X, Y)
    x_t=clf.transform(X)
    #t=np.zeros(8)
    #t[0]=1.0
    #print(clf.transform(t))
    cor1=[x_t[i][0] for i in xrange(len(x_t)) ]
    cor2=[x_t[i][1] for i in xrange(len(x_t)) ]
    #cor3=[x_t[i][1] for i in xrange(len(x_t)) ]

    #print(clf.xbar_)
    print(clf.scalings_)    
    return cor1,cor2#,cor3
  
def apply_MDA(X):    
    x_t,n=reduction.mdaReduction(X,dim=2)
    cor1=[x_t[i][0] for i in xrange(len(x_t)) ]
    cor2=[x_t[i][1] for i in xrange(len(x_t)) ]
    return cor1,cor2  
    
def apply_tsne(X):    
    x_t,n=reduction.tsneReduction(X,dim=2)
    cor1=[x_t[i][0] for i in xrange(len(x_t)) ]
    cor2=[x_t[i][1] for i in xrange(len(x_t)) ]
    return cor1,cor2
    

#showLDA()
showDataset()
#dataset=arff.readArffDataset("innovation.arff")
#print(apply_PCA(dataset.data))
#visualize.showReduction(dataset,reduction.pcaReduction,"3_12_8",False)