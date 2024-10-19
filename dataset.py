import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing

class Dataset(object):
    def __init__(self,X,y=None):
        self.X=X
        self.y=y

    def __len__(self):
        return len(self.y)

    def dim(self):
        return self.X.shape[1]

    def n_cats(self):
        return int(max(self.y))+1

    def get_cat(self,i):
    	return self.X[self.y==i]
		
    def __call__(self,fun):
        return Dataset(X=fun(self.X),
                       y=self.y)

    def split(self,train_index,test_index):
        X_train=self.X[train_index]
        y_train=self.y[train_index]
        X_test=self.X[test_index]
        y_test=self.y[test_index]
        return (X_train,y_train),(X_test,y_test)        

    def eval(self,train_index,test_index,clf):
        (X_train,y_train),(X_test,y_test)=self.split(train_index,test_index)
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)
        return y_pred,y_test

    def labeled(self):
        return not (self.y is None)

    def min(self):
        return np.amin(self.X,axis=0)

    def max(self):
        return np.amax(self.X,axis=0)

    def class_weight(self):
        params={}
        for i in range(self.n_cats()):
            size_i= sum((self.y==i).astype(int))
            params[i]= 1.0/size_i
        Z= sum(list(params.values()))
        for i in params:
            params[i]= params[i]/Z
        return params

class Clustering(object):
    def __init__(self,dataset,cls_indices):
        self.dataset=dataset
        self.cls_indices=cls_indices

    def n_cluster(self):
        return int(max(self.cls_indices))+1

    def hist(self):
        hist=np.zeros((self.n_cluster(),
                       self.dataset.n_cats()))
        for i,clf_i in enumerate(self.cls_indices):
            y_i=int(self.dataset.y[i])
            hist[clf_i][y_i]+=1
        return hist

def read_csv(in_path:str):
    if(type(in_path)==tuple):
        X,y=in_path
        return Dataset(X,y)
    df=pd.read_csv(in_path,header=None)
    raw=df.to_numpy()
    X,y=raw[:,:-1],raw[:,-1]
    X= preprocessing.RobustScaler().fit_transform(X)
    return Dataset(X,y)

def get_class_weights(y):
    params={}
    n_cats=int(max(y))+1
    for i in range(n_cats):
        size_i=(y==i).shape[0]
        params[i]= 1.0/size_i
    Z= sum(list(params.values()))
    for i in params:
        params[i]= params[i]/Z
    return params

def gini_coefficient(x):
    x=x/np.sum(x)
    return np.dot(x,x)
#    x = sorted(x)
#    n = len(x)
#    s = sum(x)
#    d = n * s
#    G = sum(xi * (n - i) for i, xi in enumerate(x))
#    return (d + s - 2 * G) / d

#def gini_coefficient(x):
#    x=np.sort(x)
#    raise Exception(x)
#    diffsum = 0
#    for i, xi in enumerate(x[:-1], 1):
#        diffsum += np.sum(np.abs(xi - x[i:]))
#    return diffsum / (len(x)**2 * np.mean(x))

if __name__ == '__main__':
    incomes = np.array([0,0,0,0,0,0,0,1000])#50, 50, 70, 70, 70, 90, 150, 150, 150, 150])
    print(gini_coefficient(incomes))
#    data=read_csv("../uci/lymphography")
#    for i in range(data.n_cats()):
#        x_i=data.get_cat(i)
#        print(x_i.shape)