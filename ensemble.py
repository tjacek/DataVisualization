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
    def __init__(self,weight_dict=None,full=True, verbose=0):
        super().__init__(full=full,
                         verbose=verbose)
        self.weight_dict=weight_dict

    def fit(self,X,y):
        self.reset()
        if(self.weight_dict==None):
            self.weight_dict=dataset.get_class_weights(y)
        size_dict=self.weight_dict.size_dict()
        n_cats=len(self.weight_dict)
        avg_size=1.0/n_cats
        for cat_i,weight_i in size_dict.items():
            if(weight_i<avg_size):
                dict_i=dataset.WeightDict(self.weight_dict.copy())
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

class CEnsembleFactory(object):
    def __init__(self,data):
        self.weight_dict=dataset.get_class_weights(data.y)

    def __call__(self):
        return ClassEnsemble(weight_dict=self.weight_dict)

class GaussEnsemble(Ensemble):
    def __init__(self,k,full=True, verbose=0):
        super().__init__(full=full,
                         verbose=verbose)
        self.k=k

	
    def fit(self,X,y):
        self.reset()
        clustering=gauss.gaussian_clustering((X,y),
			                                 alg_type="bayes",
			                                 k=self.k)
        hist=clustering.hist()
        recall=hist.recall_matrix()
        n_cats=clustering.dataset.n_cats()
        for k,recall_k in enumerate(recall.T):
            if(np.all(recall_k<0.5)):
                print(k)
                cls_k=clustering.wihout_cluster(k)
                if(cls_k.y.shape[0]>0):
                    nn_k=deep.ClfCNN(default_cats=n_cats,
                                     verbose=self.verbose)
                    print(cls_k.y.shape)
                    nn_k.fit(X=cls_k.X,y=cls_k.y)
                    self.clfs.append(nn_k)
        if(self.full):
            nn=deep.ClfCNN(default_cats=n_cats,
                verbose=self.verbose)
            nn.fit(X=clustering.dataset.X,
                   y=clustering.dataset.y)
            self.clfs.append(nn)

class GEnsembleFactory(object):
    def __init__(self,data):
        self.data=data
        self.k=None

    def __call__(self):
        if(self.k is None):
            _,k=gauss.good_of_fit(in_path=self.data,
        	                      alg_type="bayes",
        	                      show=False)
            self.k=k
        return 	GaussEnsemble(k=self.k)

@utils.elapsed_time
def compare_ensemble(in_path,deep_ens=None,single=True,verbose=0):
    data=dataset.read_csv(in_path)
    if(deep_ens is None):
        deep_ens=GaussEnsemble(k=3,verbose=verbose)
    def helper(split):
        deep_ens.reset()
        result_ens=split.eval(data,deep_ens)
        nn=deep.ClfCNN(verbose=verbose)
        result_nn=split.eval(data,nn)
        print(f"n_clf{len(deep_ens)}")
        return result_nn,result_ens
    protocol=exp.UnaggrSplit(n_splits=10,
    	                     n_repeats=1)
    splits=protocol.get_split(data)
    if(single):
        return helper(splits[0])
    else:
        results=[helper(split_i) for split_i in splits]
        partial_nn,partial_ens=list(zip(*results))
        result_nn=dataset.unify_results(partial_nn)
        result_ens=dataset.unify_results(partial_ens)
        return result_nn,result_ens

if __name__ == '__main__':
    deep_ens=ClassEnsemble()
    nn,ens=compare_ensemble("../uci/cleveland",
    	                    deep_ens=None,#deep_ens,
    	                    single=False,
    	                    verbose=0)
    nn.report()
    ens.report()
    print(nn.get_acc())
    print(ens.get_acc())
    print(nn.get_balanced())
    print(ens.get_balanced())
