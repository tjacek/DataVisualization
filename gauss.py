import numpy as np
from sklearn.mixture import GaussianMixture,BayesianGaussianMixture
import dataset,visualize

class BasicMixture(object):
    def __init__(self,criterion="bic"):
#        self.alg=GaussianMixture
        self.criterion=criterion
        self.alg=None

    def fit(self,X,n_components=2):
        self.alg=GaussianMixture(n_components=n_components)
        self.alg.fit(X)
    
    def compute_criterion(self,X):
        if(self.criterion=="bic"):
            return self.alg.bic(X)
        else:
            return self.alg.aic(X)

class MuliGauss(object):
    def __init__(self,mean,conv):
        self.mean=mean
        self.conv=conv
        self.inv_conv=None
        self.Z=None
    
    def __call__(self,x):
        if(self.Z is None):
            k=self.dim()
            det=np.linalg.det(self.conv)
            self.Z=(2*np.pi)**k
            self.Z=np.sqrt(self.Z*det)
        return np.exp(-0.5*self.maha(x,False))/self.Z   

    def dim(self):
        return self.mean.shape[0]

    def euclid(self,x):
        return np.linalg.norm(self.mean-x)

    def maha(self,x,sqrt=True):
        if(self.inv_conv is None):
            self.inv_conv=np.linalg.inv(self.conv)
        diff= (x-self.mean)
        m=np.matmul(self.inv_conv,diff)
        if(sqrt):
            return np.sqrt(np.dot(diff,m))
        else:
            return np.dot(diff,m)

    def eigen(self):
        eig_values=np.linalg.eigvals(self.conv)
        eig_values/= np.sum(eig_values)
        return np.round(eig_values, decimals=3)

def fit_gauss(in_path):#,verbose=True):
    data=dataset.read_csv(in_path)
    mixture=GaussianMixture(n_components=data.n_cats())
    mixture.fit(data.X)
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

def get_mixture_alg(alg_type):
    if(alg_type=="bayes"):
        return BayesianGaussianMixture
    return BasicMixture("bic")

def bic(alg, X):
    n_effect_comp = len(np.unique(alg.predict(X)))
    _, n_features = alg.means_.shape      
    cov_params = n_effect_comp * n_features * (n_features + 1) / 2.
    mean_params = n_features * n_effect_comp
    _n_parameters=int(cov_params + mean_params + n_effect_comp - 1)
    return (-2 * alg.score(X) * X.shape[0] +
                _n_parameters * np.log(X.shape[0]))

def good_of_fit(in_path,show=True,alg_type="basic"):
    data=dataset.read_csv(in_path)
    mixture_alg= get_mixture_alg(alg_type)
    criterion=[]
    for i in range(2*data.n_cats()):
#        mixture=mixture_alg(n_components=i+1)
        mixture_alg.fit(data.X,n_components=i+1)
#        raise Exception(mixture_alg.criterion(data.X))
        criterion.append(mixture_alg.compute_criterion(data.X))
    crit_max= np.abs(np.amax(criterion))
    k=np.argmin(criterion)
    norm_cri=[ crit_i/crit_max for crit_i in criterion]
    if(show):
        visualize.bar_plot(norm_cri)
    return norm_cri,k

def point_distribution(in_path,k=5,show=True):
    data=dataset.read_csv(in_path)
    mixture=GaussianMixture(n_components=k)
    mixture.fit(data.X)
    all_dist=[]
    for i in range(data.n_cats()):
        gauss_i=MuliGauss(mean=mixture.means_[i],
                          conv=mixture.covariances_[i])
        all_dist.append(gauss_i)
    hist=np.zeros((k,data.n_cats()))
    for i,x_i in enumerate(data.X):
        prob_i=[ dist_j(x_i) for dist_j in all_dist ]
        cluster_i=np.argmax(prob_i)
        y_i=int(data.y[i])
        hist[cluster_i][y_i]+=1
    visualize.stacked_bar_plot(hist,show=show)

if __name__ == '__main__':
#    visualize.HMGenerator(show_euclid)("../uci","euclid")
#    point_distribution("../uci/cleveland")
     GaussExp()("../uci","gauss")