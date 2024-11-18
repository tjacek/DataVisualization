import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import BallTree
from scipy.special import kl_div
import seaborn as sns

import dataset

def near_density(in_path,k=10):
    data=dataset.read_csv(in_path)
    tree=BallTree(data.X)
    indces= tree.query(data.X,
                       k=k+1,
                       return_distance=False)
    same_class=[]
    for i,ind_i in enumerate(indces):
        y_i=data.y[i]
        near=[ int(y_i==data.y[ind_j]) for ind_j in ind_i[1:]]
        same_class.append(np.mean(near))
    return same_class

def compute_density(data,dim,cat=None,show=False,n_steps=100):
    if(cat is None):
        x_i=data.X[:,dim]
    else:
        x_i=data.get_cat(cat)[:,dim]
    x_i=x_i.reshape(-1, 1) 
    kde = KernelDensity(kernel='gaussian', bandwidth=2).fit(x_i)
    a_max,a_min=np.max(x_i),np.min(x_i)
    delta= (a_max-a_min)/n_steps
    x_order=np.arange(n_steps)*delta
    x_order-=a_min
    log_dens= kde.score_samples(x_order.reshape(-1, 1))
    dens=np.exp(log_dens)
    if(show):
        fig, ax = plt.subplots()
        ax.plot(x_order,np.exp(log_dens))
        plt.show()
    return dens

def dim_matrix(data,cat_i=0):
    n_dims= data.dim()
    matrix=[]
    all_dens=[ compute_density(data,dim_j,cat_i) 
        for dim_j in range(n_dims)]
    for dens_i in all_dens:
        matrix.append([])
        for dens_j in all_dens:
            kl_ij=np.mean(kl_div(dens_i,dens_j))
            matrix[-1].append(kl_ij)
    show_matrix(matrix)

def cat_matrix(data,dim_i=0):
    matrix=[]
    all_dens=[ compute_density(data,dim_i,cat_j) 
        for cat_j in range(data.n_cats())]
    for dens_i in all_dens:
        matrix.append([])
        for dens_j in all_dens:
            kl_ij=np.mean(kl_div(dens_i,dens_j))
            matrix[-1].append(kl_ij)
    show_matrix(matrix)

def show_matrix(matrix):
    matrix=np.array(matrix)
    matrix[np.isnan(matrix)]=0.0
    matrix[matrix==np.inf]=0
    matrix/= np.sum(matrix)
    print(np.around(matrix,decimals=2))
    plt.figure(figsize = (10,7))
    sns.heatmap(matrix, annot=True)
    plt.show()

if __name__ == '__main__':
    near_density("../uci/wine-quality-red")
#    data=dataset.read_csv("../uci/wine-quality-red")
#    cat_matrix(data,dim_i=2)