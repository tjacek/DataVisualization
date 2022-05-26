import numpy as np

class DataDict(dict):
    def __init__(self, arg=[]):
        super(DataDict, self).__init__(arg)

    def to_dataset(self):
        names=self.keys()
        X=np.array([self[name_i] 
            for name_i in names])
        return names,X

    def transform(self,trans_fun):
        names,X=self.to_dataset()
        X_t=trans_fun(X)
        return DataDict(zip(names,X_t))
