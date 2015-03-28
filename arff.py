# -*- coding: utf-8 -*-

import re,numpy as np
from sklearn.datasets.base import Bunch

class ArffDataset(object):

    def __init__(self,attributes,instances,categories):
        self.attributes=attributes
        self.instances=instances
        self.categories=categories

    def size(self):
        return len(self.instances)
    
    def dim(self):
        return len(self.attributes)
        
    def toMatrix(self):
        toVectors=lambda inst:inst.values 
        featureVectors=map(toVectors,self.instances)
        return np.array(featureVectors)

    def getTargets(self):
        extractCategory=lambda inst:self.getIntegerCategory(inst.category)
        return np.array(map(extractCategory,self.instances))
    
    def getIntegerCategory(self,label):
        return self.categories.index(label)
        
    def toBunch(self):
        data=self.toMatrix()
        target=self.getTargets()
        return Bunch(data=data,target=target,target_names=self.categories) 
        
    def _str_(self):
        s=""
        for instance in self.instances:
            s+=str(instance)
        return s

class Instance(object):
    
    def __init__(self,values,category):        
        self.values=values
        self.category=category

    def __str__(self):
        s=""
        for cord in self.values:
            s+=str(cord)+","
        s+=self.category +"\n"   
        return s
        
def readArffDataset(filename):
    raw=open(filename).read()
    attributes,data=splitArff(raw)
    attrNames,categories=parseAttributes(attributes)
    instances=parseInstances(data)
    return ArffDataset(attributes,instances,categories)

def splitArff(raw):
    separator="@DATA"
    arff=raw.split(separator)
    return arff[0],arff[1]
    
def parseAttributes(attributes):
    isAttribute=r"@ATTRIBUTE(.)+"
    attrNames=[]
    categories=None
    for line in attributes.split("\n"):
        matchObj = re.match(isAttribute, line,re.I)
        if(matchObj):
            rows=line.split()
            name=rows[1]
            if(name!="class"):
                attrNames.append(name)
            else:
                categories=parseCategories(line)
    return attrNames,categories
    
def parseInstances(data):
    instances=[]
    toNumber= lambda s: float(s)
    for line in data.split("\n"):
        values=line.split(",")
        if(len(values)>1):
            category=values.pop()
            values=map(toNumber,values)
            instance=Instance(values,category)
            instances.append(instance)
    return instances
        
def parseCategories(line):
    categories=line.split("{")[1]
    categories=categories.replace("}", "")
    return categories.split()