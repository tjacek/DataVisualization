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
        return int(max(self.y))

    def get_cat(self,i):
    	return self.X[self.y==i]
		
    def __call__(self,fun):
        return Dataset(X=fun(self.X),
                       y=self.y)

    def eval(self,train_index,test_index,clf):
        X_train=self.X[train_index]
        y_train=self.y[train_index]
        X_test=self.X[test_index]
        y_test=self.y[test_index]
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)
        return y_pred,y_test

    def labeled(self):
        return not (self.y is None)

    def min(self):
        return np.amin(self.X,axis=0)

    def max(self):
        return np.amax(self.X,axis=0)

def read_csv(in_path:str):
    df=pd.read_csv(in_path)
    raw=df.to_numpy()
    X,y=raw[:,:-1],raw[:,-1]
    X= preprocessing.RobustScaler().fit_transform(X)
    return Dataset(X,y)

def get_pca(X,y=None,
            n_components=None,
            verbose=False):
    if(type(X)==Dataset):
        X,y=X.X,X.y
    pca = PCA(n_components=n_components)
    pca.fit(X)
    if(verbose):
        print(pca.explained_variance_ratio_)
    return Dataset(X=pca.transform(X),
                y=y)

if __name__ == '__main__':
    data=read_csv("uci/cleveland")
    for i in range(data.n_cats()):
        x_i=data.get_cat(i)
        get_pca(x_i)