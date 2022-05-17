#import arff
#import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
#from numpy import linalg as LA
#from sklearn.lda import LDA
import pandas as pd 

def show_data(in_path):
    df = read_data(in_path)
    df = remove_outliners(df)
    names,X= split_frames(df)
    pca = PCA(n_components=2)
    x_t=pca.fit(X).transform(X)
    print(pca.components_)
    print(pca.explained_variance_ratio_)
    print(x_t)
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

def remove_outliners(df):
    id_name=df.columns[0]
    col_names=list(df.columns[1:])
    extr={ col_i:(df[col_i].max(),df[col_i].min()) 
        for col_i in col_names}
    outliners=set()
    for name_i,(max_i,min_i) in extr.items():
        result=df[(df[name_i]==max_i)]  #df.query(f"{name_i}=={max_i}")
        outliners.update( list(result[id_name]) )
    for out_i in outliners:
        df = df[df[id_name] != out_i]
    return df

#def showLDA():
#    dataset=arff.readArffDataset("innovation_.arff")
#    x,y,z=apply_LDA(dataset.data,dataset.target)
#    countryVisual3D(x,y,z,dataset.target)
    
#def countryVisual(x,y,cat):
#    fig, ax = plt.subplots()
#    c_p=[colors[i] for i in cat ]
#    ax.scatter(x, y,c=c_p)
#    for i,txt in enumerate(ccodes):
#        ax.annotate(txt, (x[i],y[i]))

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
  
#def apply_MDA(X):    
#    x_t,n=reduction.mdaReduction(X,dim=2)
#    cor1=[x_t[i][0] for i in xrange(len(x_t)) ]
#    cor2=[x_t[i][1] for i in xrange(len(x_t)) ]
#    return cor1,cor2  
    
#def apply_tsne(X):    
#    x_t,n=reduction.tsneReduction(X,dim=2)
#    cor1=[x_t[i][0] for i in xrange(len(x_t)) ]
#    cor2=[x_t[i][1] for i in xrange(len(x_t)) ]
#    return cor1,cor2
    
show_data(["dcss.txt","dcss2.txt","dcss3.txt"])