from sklearn.mixture import GaussianMixture
import dataset

def fit_gauss(in_path,verbose=True):
    if(type(in_path)==str):
        data=dataset.read_csv(in_path)
    else:
        data=in_path
    mixture=GaussianMixture(n_components=data.n_cats())
    mixture.fit(data.X)
    print(dir(mixture))
    if(verbose):
       print(f"bic:{mixture.bic(data.X)}")
       print(f"aic:{mixture.aic(data.X)}")
    return mixture

if __name__ == '__main__':
    fit_gauss("uci/cleveland")