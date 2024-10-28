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

    def reset(self):
        self.clfs=[]	

class ClassEnsemble(Ensemble):
    def fit(self,X,y):
    	weight_dict=dataset.get_class_weights(y)
    	n_cats=len(weight_dict)
    	raise Exception(weight_dict)

class GaussEnsemble(Ensemble):
	def __init__(self,k,full=True, verbose=0):
		super().__init__()
		self.k=k
		self.full=full
		self.verbose=verbose 
	
	def fit(self,X,y):
		clustering=gauss.gaussian_clustering((X,y),k=self.k)
		hist=clustering.hist()
#		print(hist.arr)
		recall=hist.recall_matrix()
		n_cats=clustering.dataset.n_cats()
		for k,recall_k in enumerate(recall.T):#clustering.n_clusters()):
			if(np.all(recall_k<0.5)):
				cls_k=clustering.wihout_cluster(k)
				nn_k=deep.ClfCNN(default_cats=n_cats,
								 verbose=self.verbose)
				nn_k.fit(X=cls_k.X,y=cls_k.y)
				self.clfs.append(nn_k)
		if(self.full):
			nn=deep.ClfCNN(default_cats=n_cats,
						   verbose=self.verbose)
			nn.fit(X=clustering.dataset.X,
				   y=clustering.dataset.y)
			self.clfs.append(nn)

@utils.elapsed_time
def compare_ensemble(in_path,deep_ens=None,single=True,verbose=0):
    data=dataset.read_csv(in_path)
    if(deep_ens is None):
        deep_ens=GaussEnsemble(k=3,verbose=verbose)
    def helper(train,test):
        print("OK")
        deep_ens.reset()
        nn=deep.ClfCNN(verbose=verbose)
        result_ens=data.eval(train,test,deep_ens)
        result_nn=data.eval(train,test,nn)
        return result_nn,result_ens
    gen=exp.simple_split(data,n_splits=10)
    if(single):
        train,test=next(gen)
        return helper(train,test)
    else:
        results=[helper(train_i,test_i) for train_i,test_i in gen]
        partial_nn,partial_ens=list(zip(*results))
        result_nn=dataset.unify_results(partial_nn)
        result_ens=dataset.unify_results(partial_ens)
        return result_nn,result_ens

if __name__ == '__main__':
    deep_ens=None#ClassEnsemble()
    nn,ens=compare_ensemble("uci/cleveland",
    	                    deep_ens=deep_ens,
    	                    single=False,
    	                    verbose=0)
    print(nn.get_acc())
    print(ens.get_acc())
    print(nn.get_balanced())
    print(ens.get_balanced())