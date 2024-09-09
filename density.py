import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import dataset

def show_density(data,dim,cat=None):
    if(cat is None):
        x_i=data.X[:,dim]
    else:
        x_i=data.get_cat(cat)[:,dim]
    x_i=x_i.reshape(-1, 1) 
    kde = KernelDensity(kernel='gaussian', bandwidth=2).fit(x_i)
    x_order= np.sort(np.unique(x_i))
    log_dens= kde.score_samples(x_order.reshape(-1, 1))
    fig, ax = plt.subplots()
    ax.plot(x_order,np.exp(log_dens))
    plt.show()

if __name__ == '__main__':
    data=dataset.read_csv("uci/cleveland")
    show_density(data,0,2)