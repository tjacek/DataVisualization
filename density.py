import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy.special import kl_div
import dataset

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
#    x_order= np.sort(np.unique(x_i))
    log_dens= kde.score_samples(x_order.reshape(-1, 1))
    dens=np.exp(log_dens)
#    dens=log_dens
    if(show):
        fig, ax = plt.subplots()
        ax.plot(x_order,np.exp(log_dens))
        plt.show()
    return dens

def kl_matrix(data,cat_i=0):
    n_dims= data.dim()
    matrix=[]
    all_dens=[ compute_density(data,dim_j,cat_i) 
        for dim_j in range(n_dims)]
    print([d.shape for d in all_dens])

    for dens_i in all_dens:
        matrix.append([])
        for dens_j in all_dens:
            kl_ij=np.mean(kl_div(dens_i,dens_j))
            matrix[-1].append(kl_ij)
    matrix=np.array(matrix)
    print(np.around(matrix,decimals=2))
    
if __name__ == '__main__':
    data=dataset.read_csv("uci/cleveland")
    kl_matrix(data)