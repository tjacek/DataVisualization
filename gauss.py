from sklearn.mixture import GaussianMixture
import dataset

def fit_gauss(in_path,verbose=True):
    data=dataset.read_csv(in_path)
    mixture=GaussianMixture(n_components=data.n_cats())
    mixture.fit(data.X)
    if(verbose):
       print(f"bic:{mixture.bic(data.X)}")
       print(f"aic:{mixture.aic(data.X)}")
    return mixture

fit_gauss("uci/cleveland")