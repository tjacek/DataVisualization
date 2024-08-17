import pandas as pd
from sklearn.decomposition import PCA

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

def read_csv(in_path:str):
    df=pd.read_csv(in_path)
    raw=df.to_numpy()
    X,y=raw[:,:-1],raw[:,-1]
    return Dataset(X,y)

def get_pca(X,y=None):
    pca = PCA()#n_components=2)
    pca.fit(X)
    print(pca.explained_variance_ratio_)
    return Dataset(X=pca.transform(X),
                y=y)
if __name__ == '__main__':
    data=read_csv("uci/cleveland")
    for i in range(data.n_cats()):
        x_i=data.get_cat(i)
        get_pca(x_i)
