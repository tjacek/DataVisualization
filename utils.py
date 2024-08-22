import pandas as pd
from sklearn.decomposition import PCA
from functools import wraps

class Dataset(object):
    def __init__(self,X,y):
        self.X=X
        self.y = y

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

def read_csv(in_path:str):
    df=pd.read_csv(in_path)
    raw=df.to_numpy()
    X,y=raw[:,:-1],raw[:,-1]
    return Dataset(X,y)

def make_dir(path):
    if(not os.path.isdir(path)):
        os.mkdir(path)

def get_pca(X,y=None):
    pca = PCA()#n_components=2)
    pca.fit(X)
    print(pca.explained_variance_ratio_)
    return Dataset(X=pca.transform(X),
                y=y)


class DirFun(object):
    def __init__(self,dir_args=None):
        if(dir_args is None):
            dir_args=[("in_path",0)]
        self.dir_args=dir_args

    def __call__(self, fun):
        @wraps(fun)
        def decor_fun(*args, **kwargs):
            in_path=self.get_input(*args, **kwargs)
            for in_i in top_files(in_path):
                fun(*args, **kwargs)
        return decor_fun
        
    def get_input(self,*args, **kwargs):
        name,i=self.dir_args[0]
        if(name in kwargs):
            return kwargs[name]
        return args[0]

if __name__ == '__main__':
    data=read_csv("uci/cleveland")
    for i in range(data.n_cats()):
        x_i=data.get_cat(i)
        get_pca(x_i)
