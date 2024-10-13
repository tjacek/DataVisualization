import numpy as np
from sklearn.mixture import GaussianMixture
import dataset,visualize

class MuliGauss(object):
    def __init__(self,mean,conv):
        self.mean=mean
        self.conv=conv
        self.inv_conv=None

    def euclid(self,x):
        return np.linalg.norm(self.mean-x)

    def maha(self,x):
        if(self.inv_conv is None):
            self.inv_conv=np.linalg.inv(self.conv)
        diff= (x-self.mean)
        m=np.matmul(self.inv_conv,diff)
        return np.sqrt(np.dot(diff,m))

    def eigen(self):
        eig_values=np.linalg.eigvals(self.conv)
        eig_values/= np.sum(eig_values)
        return np.round(eig_values, decimals=3)

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
        print(x_i.shape)
        alg_i=GaussianMixture(n_components=1)
        alg_i.fit(x_i)
        gauss_i=MuliGauss(mean=np.squeeze(alg_i.means_,0),
                          conv=np.squeeze(alg_i.covariances_,0))
        all_dist.append(gauss_i)
    return all_dist

def show_euclid(in_path,show=False):
    all_dist=cat_gauss(in_path)
    matrix=[[  dist_j.euclid(dist_i.mean)
               for dist_j in all_dist]
                   for dist_i in all_dist] 
    if(show):
        visualize.show_matrix(matrix)
    return matrix

def show_maha(in_path,show=False):
    all_dist=cat_gauss(in_path)
    matrix=[[  dist_j.maha(dist_i.mean)
               for dist_j in all_dist]
                   for dist_i in all_dist] 
    if(show):
        visualize.show_matrix(matrix)
    return matrix

def eigen_gauss(in_path):
    all_dist=cat_gauss(in_path)
    matrix=[ dist_i.eigen() for dist_i in all_dist]
    visualize.show_matrix(matrix)

def good_of_fit(in_path):
    data=dataset.read_csv(in_path)
    criterion=[]
    for i in range(2*data.n_cats()):
        mixture=GaussianMixture(n_components=i+1)
        mixture.fit(data.X)
        criterion.append( mixture.aic(data.X))
    crit_max=np.amax(criterion)
    norm_cri=[ crit_i/crit_max for crit_i in criterion]
    print(norm_cri)

if __name__ == '__main__':
#    visualize.HMGenerator(show_euclid)("../uci","euclid")
    good_of_fit("../uci/cleveland")