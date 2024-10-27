import numpy as np
import deep,gauss,dataset,exp,utils

class Ensemble(object):
    def __init__(self):
        self.clfs=[]

    def predict(self,X):
        votes=[clf_i.predict_proba(X)
                for clf_i in self.clfs]
        votes=np.array(votes)
        votes=np.sum(votes,axis=0)
        return np.argmax(votes,axis=1) 

class GaussEnsemble(Ensemble):
	def __init__(self,k, verbose=0):
		super().__init__()
		self.k=k
		self.verbose=verbose 
	
	def fit(self,X,y):
		clustering=gauss.gaussian_clustering((X,y),k=self.k)
		hist=clustering.hist()
		print(hist.arr)
		raise Exception(hist.recall_matrix())
		n_cats=clustering.dataset.n_cats()
		for k in range(clustering.n_clusters()):
			cls_k=clustering.wihout_cluster(k)
			nn_k=deep.ClfCNN(default_cats=n_cats,
				             verbose=self.verbose)
			nn_k.fit(X=cls_k.X,y=cls_k.y)
			self.clfs.append(nn_k)

@utils.elapsed_time
def compare_ensemble(in_path,deep_ens=None,verbose=0):
    data=dataset.read_csv(in_path)
    if(deep_ens is None):
        deep_ens=GaussEnsemble(k=3,verbose=verbose)

    nn=deep.ClfCNN(verbose=verbose)
    gen=exp.simple_split(data,n_splits=10)
    train,test=next(gen)
#    result_nn=data.eval(train,test,nn)
    result_ens=data.eval(train,test,deep_ens)
    return result_nn,result_ens

if __name__ == '__main__':
    nn,ens=compare_ensemble("uci/cleveland",
    	                    verbose=0)
    nn.report()
    ens.report()
#    print(nn.get_acc())
#    print(ens.get_acc())