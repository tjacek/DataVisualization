# -*- coding: utf-8 -*-

import re

class Dataset(object):
    def __init__(self,attributes,instances,categories):
        self.attributes=attributes
        self.instances=instances
        self.categories=categories

class Instance(object):
    def __init__(self,values,category):        
        self.values=values
        self.category=category
    
def readArffDataset(filename):
    raw=open(filename).read()
    attributes,data=splitArff(raw)
    attrNames,categories=parseAttributes(attributes)
    instances=parseInstances(data)
    return Dataset(attributes,instances,categories)

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
        category=values.pop()
        values=map(toNumber,values)
        instance=Instance(values,category)
        instances.append(instance)
    return instances
        
def parseCategories(line):
    categories=line.split("{")[1]
    categories=categories.replace("}", "")
    return categories.split()

readArffDataset("C:/Users/user/Desktop/kwolek/output/5_12_8_0.arff")