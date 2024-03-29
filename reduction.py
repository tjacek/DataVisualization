import sklearn.ensemble
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn import manifold
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

def pca_transform(X,n_dim=2):
    scaler=StandardScaler().fit(X)
    X=scaler.transform(X)
    pca = PCA(n_components=n_dim)
    x_t=pca.fit(X).transform(X)
    print(pca.components_)
    print(pca.explained_variance_ratio_)
    return x_t,pca

def mda_transform(X):    
    embedding = MDS(n_components=2)
    return embedding.fit_transform(X)

def lda_transform(X,y):
    clf = LinearDiscriminantAnalysis(n_components=2)
    X_t=clf.fit(X,y).transform(X)
    if(X_t.shape[-1]==1):
        zero_col=np.zeros(X_t.shape)
        X_t=np.concatenate([X_t,zero_col],axis=1)
    return X_t,clf

def tsne_transform(X,n_components=2):    
    tsne = TSNE(n_components=n_components, init='pca', random_state=0)
    X_t=tsne.fit_transform(X)
    return X_t,None

def lle_transform(X,dim=2,n_neighbors=5):
    clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=dim,
                                      method='standard')
    return clf.fit_transform(X),clf

def spectral_transform(X,n_components=2):
    embedder = manifold.SpectralEmbedding(n_components=n_components, 
        random_state=0,eigen_solver="arpack")
    return embedder.fit_transform(X),embedder

def ensemble_transform(X,n_components=2):
    hasher = sklearn.ensemble.RandomTreesEmbedding(n_estimators=200, random_state=0,
                                       max_depth=5)
    X_transformed = hasher.fit_transform(X)
    pca = TruncatedSVD(n_components=n_components)
    X_reduced = pca.fit_transform(X_transformed)
    return X_reduced,None