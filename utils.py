import pandas as pd
from sklearn.decomposition import PCA

def read_csv(in_path:str):
    df=pd.read_csv(in_path)
    X=df.to_numpy()
    pca = PCA()#n_components=2)
    pca.fit(X)
    print(pca.explained_variance_ratio_)
#    print(X.shape)

if __name__ == '__main__':
	read_csv("uci/cleveland")
