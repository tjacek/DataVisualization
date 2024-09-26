import sklearn.ensemble
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn import manifold
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import dataset

def spectral_transform(data,n_components=2):
    embedder = manifold.SpectralEmbedding(n_components=n_components, 
                                          random_state=0,
                                          eigen_solver="arpack")
    new_X=embedder.fit_transform(data.X)
    return dataset.Dataset(X=new_X,
                           y=data.y)

def lda_transform(data,n_components=2):
    clf = LinearDiscriminantAnalysis(n_components=n_components)
    X_t=clf.fit(data.X,data.y).transform(data.X)
    if(X_t.shape[-1]==1):
        zero_col=np.zeros(X_t.shape)
        X_t=np.concatenate([X_t,zero_col],axis=1)
    return dataset.Dataset(X=X_t,
                           y=data.y)

def lle_transform(data,n_components=2,n_neighbors=5):
    clf = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, 
                                          n_components=n_components,
                                          method='standard')
    return dataset.Dataset(X=clf.fit_transform(data.X),
                           y=data.y)

def mda_transform(data,n_components=2):    
    embedding = MDS(n_components=n_components)
    return dataset.Dataset(X=embedding.fit_transform(data.X),
                           y=data.y)

def tsne_transform(data,n_components=2):    
    tsne = TSNE(n_components=n_components, 
                init='pca', 
                random_state=0)
    X_t=tsne.fit_transform(data.X)
    return dataset.Dataset(X=X_t,
                           y=data.y)

def ensemble_transform(data,n_components=2):
    hasher = sklearn.ensemble.RandomTreesEmbedding(n_estimators=200, 
                                                   random_state=0,
                                                   max_depth=5)
    X_transformed = hasher.fit_transform(data.X)
    pca = TruncatedSVD(n_components=n_components)
    X_reduced = pca.fit_transform(X_transformed)
    return dataset.Dataset(X=X_reduced,
                           y=data.y)

def pca_transform(data,
            n_components=None,
            verbose=False):
    pca = PCA(n_components=n_components)
    pca.fit(data.X)
    if(verbose):
        print(pca.explained_variance_ratio_)
    return Dataset(X=pca.transform(X),
                   y=data.y)

if __name__ == '__main__':
    data=dataset.read_csv("uci/cleveland")
    spectral_transform(data,n_components=2)