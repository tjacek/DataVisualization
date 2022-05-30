import os
import json
import numpy as np

class DataDict(dict):
    def __init__(self, arg=[],supervised=False):
        super(DataDict, self).__init__(arg)
        self.supervised=supervised

    def __setitem__(self, key, value):
        if(type(key)==str):
            key=files.Name(key)
        super(DataDict, self).__setitem__(key, value)

    def to_dataset(self):
        names=self.keys()
        X=np.array([self[name_i] 
            for name_i in names])
        if(self.supervised):
            y=self.get_cat()
            return names,X,y
        return names,X,None

    def transform(self,trans_fun):
        names,X,y=self.to_dataset()
        X_t=trans_fun(X)
        return DataDict(zip(names,X_t),supervised=self.supervised)

    def get_cat(self):
        if(self.supervised):
            return np.array([key_i.get_cat() 
                    for key_i in self.keys()])
        else:
            return []

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
        return DataDict( data_dict,supervised=True)
        
    raise Exception(f"{in_path} is not directory" )