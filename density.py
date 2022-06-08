import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

def show_density(col_i):
    col_i= col_i[col_i!='?']
    X=col_i.to_numpy().astype(float)
    X = X[~np.isnan(X)]
    X=X.reshape(-1, 1) 
    kde = KernelDensity(kernel='gaussian', bandwidth=2).fit(X)
    x_order= np.sort(np.unique(X))
    log_dens= kde.score_samples(x_order.reshape(-1, 1))
    fig, ax = plt.subplots()
    print(list(zip(x_order,np.exp(log_dens))))
    ax.plot(x_order,np.exp(log_dens))
    plt.show()

df= pd.read_csv("NF.csv")#,sep='\s+')
show_density(df['Data ur.']) 