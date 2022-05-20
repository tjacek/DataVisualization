#import arff
#import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
#from numpy import linalg as LA
from sklearn.manifold import MDS
import pandas as pd 

def show_data(in_path):
    df = read_data(in_path)
#    df = remove_outliners(df,max_cond)
    df = remove_outliners(df,min_cond)
    names,X= split_frames(df)
    x_t= pca_transform(X)
    plot(names,x_t)

def plot(names,data):
    fig, ax = plt.subplots()
    ax.scatter(data[:,0],data[:,1])
    for i,txt in enumerate(names):
        ax.annotate(txt,data[i])
    plt.show()

def read_data(in_path):
    if(type(in_path)==list):
        all_dfs=[ pd.read_csv(path_i,sep='\s+') 
                for path_i in in_path]
        main_col=all_dfs[0].columns[0]
        df=all_dfs[0]
        for df_i in all_dfs[1:]:
            df = pd.merge(left=df,right=df_i, left_on=main_col, right_on=main_col)
        print(df)
    else:
         df = pd.read_csv(in_path,sep='\s+')
    return df

def split_frames(df):
    vector=df.to_numpy()
    X=vector[:,1:]
    names=vector[:,0]
    return names,X

def remove_outliners(df,cond=None):
    if(cond is None):
        cond=max_cond
    id_name=df.columns[0]
    col_names=list(df.columns[1:])
    outliners=set()
    for col_i in col_names:
        result= df[cond(df[col_i])]
        outliners.update(list(result[id_name]))
        print(outliners)
    print(outliners)
    for out_i in outliners:
        df = df[df[id_name] != out_i]
    return df

def max_cond(df_col):
    return (df_col  ==df_col.max()) 

def min_cond(df_col):
    return (df_col  ==df_col.min()) 

def pca_transform(X,n_dim=2):
    pca = PCA(n_components=n_dim)
    x_t=pca.fit(X).transform(X)
    print(pca.components_)
    print(pca.explained_variance_ratio_)
    return x_t

def mda_transform(X):    
    embedding = MDS(n_components=2)
    return embedding.fit_transform(X)

#def showLDA():
#    dataset=arff.readArffDataset("innovation_.arff")
#    x,y,z=apply_LDA(dataset.data,dataset.target)
#    countryVisual3D(x,y,z,dataset.target)

#def countryVisual3D(x,y,z,cat):
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
#    c_p=[colors[i] for i in cat ]
#    ax.scatter(x,y,z,c=c_p)
#    for i,txt in enumerate(ccodes):
#        ax.annotate(txt, (x[i],y[i],z[i]))

#def apply_LDA(X,Y):
#    clf = LDA(n_components=2)
#    clf.fit(X, Y)
#    x_t=clf.transform(X)
#    cor1=[x_t[i][0] for i in xrange(len(x_t)) ]
#    cor2=[x_t[i][1] for i in xrange(len(x_t)) ]
#    print(clf.scalings_)    
#    return cor1,cor2#,cor3  
    
#def apply_tsne(X):    
#    x_t,n=reduction.tsneReduction(X,dim=2)
#    cor1=[x_t[i][0] for i in xrange(len(x_t)) ]
#    cor2=[x_t[i][1] for i in xrange(len(x_t)) ]
#    return cor1,cor2
    
show_data(["dcss.txt","dcss2.txt","dcss3.txt"])