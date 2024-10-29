import numpy as np
import deep,gauss,dataset,exp,utils

class Ensemble(object):
    def __init__(self,full=True, verbose=0):
        self.clfs=[]
        self.full=full
        self.verbose=verbose

    def __len__(self):
        return len(self.clfs)	

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
        size_dict=weight_dict.size_dict()
        n_cats=len(weight_dict)
        avg_size=1.0/n_cats
        for cat_i,weight_i in size_dict.items():
            if(weight_i<avg_size):
                dict_i=dataset.WeightDict(weight_dict.copy())
                dict_i[cat_i]= dict_i[cat_i]*(n_cats/2)
                nn_k=deep.ClfCNN(default_cats=n_cats,
								 default_weights=dict_i,
								 verbose=self.verbose)
                nn_k.fit(X=X,y=y)
                self.clfs.append(nn_k)
        if(self.full):
            nn=deep.ClfCNN(default_cats=n_cats,
                           verbose=self.verbose)
            nn.fit(X=X,y=y)
            self.clfs.append(nn)
        return self

class GaussEnsemble(Ensemble):
	def __init__(self,k,full=True, verbose=0):
		super().__init__(full=full,
			             verbose=verbose)
		self.k=k

	
	def fit(self,X,y):
		clustering=gauss.gaussian_clustering((X,y),k=self.k)
		hist=clustering.hist()
#		print(hist.arr)
		recall=hist.recall_matrix()
		n_cats=clustering.dataset.n_cats()
		for k,recall_k in enumerate(recall.T):
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

class GEnsembleFactory(object):
    def __call__(self,data):
        	

@utils.elapsed_time
def compare_ensemble(in_path,deep_ens=None,single=True,verbose=0):
    data=dataset.read_csv(in_path)
    if(deep_ens is None):
        deep_ens=GaussEnsemble(k=3,verbose=verbose)
    def helper(train,test):
        deep_ens.reset()
        nn=deep.ClfCNN(verbose=verbose)
        result_ens=data.eval(train,test,deep_ens)
        result_nn=data.eval(train,test,nn)
        print(f"n_clf{len(deep_ens)}")
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
    deep_ens=ClassEnsemble()
    nn,ens=compare_ensemble("uci/cleveland",
    	                    deep_ens=deep_ens,
    	                    single=False,
    	                    verbose=0)
    nn.report()
    ens.report()
    print(nn.get_acc())
    print(ens.get_acc())
    print(nn.get_balanced())
    print(ens.get_balanced())