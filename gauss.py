import numpy as np
from sklearn.mixture import GaussianMixture
import dataset,visualize

class MuliGauss(object):
    def __init__(self,mean,conv):
        self.mean=mean
        self.conv=conv

    def euclid(self,x):
        return np.linalg.norm(self.mean-x)

def fit_gauss(in_path,verbose=True):
    data=dataset.read_csv(in_path)
    mixture=GaussianMixture(n_components=data.n_cats())
    mixture.fit(data.X)
    print(dir(mixture))
    if(verbose):
       print(f"bic:{mixture.bic(data.X)}")
       print(f"aic:{mixture.aic(data.X)}")
    return mixture

def cat_gauss(in_path):
    data=dataset.read_csv(in_path)
    all_dist=[]
    for i in range(data.n_cats()):
        x_i=data.get_cat(i)
        alg_i=GaussianMixture(n_components=1)
        alg_i.fit(x_i)
        gauss_i=MuliGauss(mean=np.squeeze(alg_i.means_,0),
                          conv=np.squeeze(alg_i.covariances_,0))
        all_dist.append(gauss_i)
    return all_dist

def show_euclid(in_path):
    all_dist=cat_gauss(in_path)
    matrix=[[  dist_j.euclid(dist_i.mean)
               for dist_j in all_dist]
                   for dist_i in all_dist] 
    visualize.show_matrix(matrix)

if __name__ == '__main__':
    show_euclid("uci/cleveland")