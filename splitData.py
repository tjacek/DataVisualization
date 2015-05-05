# -*- coding: utf-8 -*-
"""
Created on Tue May 05 20:26:27 2015

@author: TP
"""

import shutil 
from os import listdir
from os.path import isfile,isdir, join

def onlyFiles(mypath):
    return [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
   
def onlyDirs(mypath):
    return [ f for f in listdir(mypath) if isdir(join(mypath,f)) ]

def splitDataset(source,dst):
    for category in onlyDirs(source):
        categoryPath= source +"/"+category
        for instance in onlyFiles(categoryPath):
            if(isOdd(instance)):
                srcPath=categoryPath+"/"+instance
                dstPath=dst+"/"+ category+"/"+instance
                shutil.copyfile(srcPath, dstPath)
                print(dstPath)

def isOdd(instance):
    xN=instance.split("_")[1]
    N=int(xN.replace("s",""))
    return (N % 2)!=0

source='C:/Users/TP/Desktop/doktoranckie/Dataset/Full'
dst='C:/Users/TP/Desktop/doktoranckie/Dataset/Test'
splitDataset(source,dst)