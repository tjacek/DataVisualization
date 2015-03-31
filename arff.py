# -*- coding: utf-8 -*-

import re,numpy as np
from sklearn import feature_selection,preprocessing

class ArffDataset(object):

    def __init__(self,attributes,instances,categories,scaled=True):
        self.attributes=attributes
        data,target=getMatrixDataset(instances,categories)
        self.data=data
        self.target=target
        self.target_names=categories
        if(scaled):
            self.preprocess()
    
    def preprocess(self):
        selector = feature_selection.VarianceThreshold()
        self.data=selector.fit_transform(self.data)
        self.data=preprocessing.scale(self.data)        
        
    def size(self):
        return len(self.target)
    
    def dim(self):
        return len(self.attributes)        

    def getCategory(self,i):
        index=self.target[i]
        return self.target_names[index]
        
    def applyReduction(self,newDim,reduction):
        #print(self.data)
        self.data,reduName=reduction(self.data,newDim)
        self.attributes=[]
        for i in range(newDim):
            self.attributes.append(reduName+"_"+str(i))
        return reduName
        
    def __str__(self):
        s="@RELATION scikit \n"
        for atr in self.attributes:
            s+="@ATTRIBUTE "+ atr+" numeric\n"
        s+="@ATTRIBUTE class {" + self.categories() +"}\n"  
        s+='@Data\n'
        for i in range(self.size()):
            sample=self.data[i]
            for cord in sample:
                s+= str(cord) +','
            s+= self.getCategory(i)+'\n'    
        return s
   
    def categories(self):
        s=""
        for cat in self.target_names:
            s+=cat+' '
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

def getMatrixDataset(instances,categories):
    dataset=toMatrix(instances)
    targets=getTargets(instances,categories)
    return dataset,targets

def toMatrix(instances):
    toVectors=lambda inst:inst.values 
    featureVectors=map(toVectors,instances)
    return np.array(featureVectors)

def getTargets(instances,categories):
    getIntegerCategory=lambda label:categories.index(label)
    extractCategory=lambda inst:getIntegerCategory(inst.category)
    return np.array(map(extractCategory,instances))
    
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