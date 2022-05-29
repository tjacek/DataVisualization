import numpy as np

class DataDict(dict):
    def __init__(self, arg=[]):
        super(DataDict, self).__init__(arg)

    def __setitem__(self, key, value):
        if(type(key)==str):
            key=files.Name(key)
        super(DataDict, self).__setitem__(key, value)

    def to_dataset(self):
        names=self.keys()
        X=np.array([self[name_i] 
            for name_i in names])
        return names,X

    def transform(self,trans_fun):
        names,X=self.to_dataset()
        X_t=trans_fun(X)
        return DataDict(zip(names,X_t))

class Name(str):
    def __new__(cls, p_string):
        return str.__new__(cls, p_string)

    def get_cat(self):
        return int(self.split('_')[1])-1

    def get_id(self):
        return self.split('_')[0]