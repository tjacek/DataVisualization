import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import BallTree
from scipy.special import kl_div
import seaborn as sns
import dataset,utils

@utils.DirFun({'in_path':0,'out_path':1})
def density_plot(in_path,out_path,k=10,all_cats=True):
    print(in_path)
    near_mean=near_density(in_path,
                           k=k,
                           all_cats=all_cats) 
    if(all_cats):
        x_order,dens= compute_density(near_mean,
                                      show=False,
                                      n_steps=100)
        fig, ax = plt.subplots()
        ax.plot(x_order,dens)
        plt.savefig(out_path)
    else:
        fig, ax = plt.subplots()
        for i,near_i in enumerate(near_mean):
            x_i,dens_i= compute_density(near_i,
                                        show=False,
                                        n_steps=100)
            ax.plot(x_i, dens_i, label=f"{i}-{len(near_i)}")
        plt.legend()
        plt.savefig(out_path)

def near_density(in_path,k=10,all_cats=True):
    data=dataset.read_csv(in_path)
    tree=BallTree(data.X)
    indces= tree.query(data.X,
                       k=k+1,
                       return_distance=False)
    same_class=[[] for _ in range(data.n_cats())]
    for i,ind_i in enumerate(indces):
        y_i=data.y[i]
        near=[ int(y_i==data.y[ind_j]) for ind_j in ind_i[1:]]
        same_class[int(y_i)].append(np.mean(near))
    if(all_cats):
        same_class=sum(same_class,[])
        return np.array(same_class)
    return [np.array(cat) for cat in same_class]

def compute_density(x,show=False,n_steps=100):
    x=x.reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(x)
    a_max,a_min=np.max(x),np.min(x)
    delta= (a_max-a_min)/n_steps
    x_order=np.arange(n_steps)*delta
    x_order-=a_min
    log_dens= kde.score_samples(x_order.reshape(-1, 1))
    dens=np.exp(log_dens)
    if(show):
        simple_plot(x_order,dens)
    return x_order,dens

def simple_plot(x_order,dens):
    fig, ax = plt.subplots()
    ax.plot(x_order,dens)
    plt.show()

#def dim_matrix(data,cat_i=0):
#    n_dims= data.dim()
#    matrix=[]
#    all_dens=[ compute_density(data,dim_j,cat_i) 
#        for dim_j in range(n_dims)]
#    for dens_i in all_dens:
#        matrix.append([])
#        for dens_j in all_dens:
#            kl_ij=np.mean(kl_div(dens_i,dens_j))
#            matrix[-1].append(kl_ij)
#    show_matrix(matrix)

#def cat_matrix(data,dim_i=0):
#    matrix=[]
#    all_dens=[ compute_density(data,dim_i,cat_j) 
#        for cat_j in range(data.n_cats())]
#    for dens_i in all_dens:
#        matrix.append([])
#        for dens_j in all_dens:
#            kl_ij=np.mean(kl_div(dens_i,dens_j))
#            matrix[-1].append(kl_ij)
#    show_matrix(matrix)

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
    density_plot("../uci","density_cat",all_cats=False)