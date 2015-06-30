# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 12:20:41 2015

@author: TP
"""
import matplotlib.pyplot as plt
from scipy import misc
import os
import numpy as np
from sklearn.lda import LDA
import matplotlib.cm as cm
import arff

def fisheraction(inp,output,cats=20):
    X,Y=readImages(inp)
    x_t=apply_LDA(X,Y)
    categories=[str(i) for i in xrange(cats)]
    attrNames="fisheraction"
    arffData=arff.saveArffNP(x_t,Y,attrNames,categories)
    afile = open(output, 'w')
    afile.write(arffData)
    afile.close()

def readImages(path):
    images=[]
    y=[]
    for f in os.listdir(path):
        fullPath=path+"/"+f
        img=misc.imread(fullPath)
        img=misc.imresize(img, 0.40)
        category=getCategory(f)
        images.append(img)
        y.append(category)    
    images=[imageToVector(img) for img in images]
    x=np.array(images)
    return x,y
    #x_t=apply_LDA(x,y)
    #print(x_t)
    #plt.imshow(vectorToImage(images[0]),cmap=cm.gist_gray)
    #plt.show()
    
def apply_LDA(X,Y):
    clf = LDA(n_components=20)
    clf.fit(X, Y)
    x_t=clf.transform(X)
    return x_t

def getCategory(name,s="cZY"):
    prefix=name.split("_")[0]
    cat=prefix.replace(s,"")
    return int(cat)

def imageToVector(img):
    return np.reshape(img,(img.size))

def vectorToImage(vect,scale=0.40):
    shape=(320*scale, 240*scale)
    return np.reshape(vect,shape)
    
def splitData(image):
    train=[]
    for i,img in enumerate(image):
        if(i % 2 ==0):        
            train.append(img)
    return train

def combineProj(xy,zx,zy,output):
    a=arff.readArffDataset(xy)
    b=arff.readArffDataset(zx)
    c=arff.readArffDataset(zy)
    instances=[]    
    for i,y_i in enumerate(a.target):
        x_1=list(a.get_x(i))
        x_2=list(b.get_x(i))
        x_3=list(c.get_x(i))
        x_n=x_1 + x_2 + x_3
        instances.append(arff.Instance(x_n,y_i))
    fisher=arff.saveArff(instances,"fisheraction",a.target_names)
    afile = open(output, 'w')
    afile.write(fisher)
    afile.close()
    
path="C:/Users/TP/Desktop/doktorancki2/Images/zy"
xy="C:/Users/TP/Desktop/doktoranckie/pyfisherXY.arff"
zx="C:/Users/TP/Desktop/doktoranckie/pyfisherZX.arff"
zy="C:/Users/TP/Desktop/doktoranckie/pyfisherZY.arff"
output="C:/Users/TP/Desktop/doktoranckie/pyfisher.arff"
combineProj(xy,zx,zy,output)
#fisheraction(path,output)