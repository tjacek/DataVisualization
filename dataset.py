import os,inspect
import json
import numpy as np

class DataDict(dict):
    def __init__(self, arg=[]):
        super(DataDict, self).__init__(arg)

    def __setitem__(self, key, value):
        if(type(key)==str):
            key=files.Name(key)
        super(DataDict, self).__setitem__(key, value)

    def names(self):
        return self.keys()

    def to_dataset(self):
        names=self.keys()
        X=np.array([self[name_i] 
            for name_i in names])
        return names,X,None

    def transform(self,trans_fun):
        names,X,y=self.to_dataset()
        X_t=trans_fun(X)
        return DataDict(zip(names,X_t))

    def get_cat(self):
        return []

class LabeledDataset(DataDict):

    def names(self):
        return [name_i.get_id() for name_i in self.keys()]

    def to_dataset(self):
        names,X,y=super(LabeledDataset,self).to_dataset()
        y=self.get_cat()
        return names,X,y

    def transform(self,trans_fun):
        names,X,y=self.to_dataset()
        if(get_arity(trans_fun)>1):
            X_t=trans_fun(X,y)
        else:
            X_t=trans_fun(X)
        return LabeledDataset(zip(names,X_t))

    def get_cat(self):
        return np.array([key_i.get_cat() 
                    for key_i in self.keys()])

class Name(str):
    def __new__(cls, p_string):
        return str.__new__(cls, p_string)

    def get_cat(self):
        return int(self.split('_')[1])

    def get_id(self):
        return self.split('_')[0]

def from_json(in_path):
    with open(in_path) as json_file:
        return json.load(json_file)

def read_class(in_path,transform=None):
    if(os.path.isdir(in_path)):
        data_dict={}
        for i,path_i in enumerate(os.listdir(in_path)):
            dict_i=from_json(f"{in_path}/{path_i}")
            for key_j,data_j in dict_i.items():
                name_ij=Name(f"{key_j}_{i}")
                data_dict[name_ij]=data_j
        if(transform):
            data_dict=transform(data_dict)
        return LabeledDataset( data_dict)
    raise Exception(f"{in_path} is not directory" )

def get_arity(func):
    desc=inspect.getargspec(func)
    if(desc[-1] is None):
        return len(desc[0])
    return len(desc[0])-len(desc[-1])