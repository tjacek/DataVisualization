import numpy as np
import deep,gauss,dataset,exp,utils
import os 


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
        self.counter=0

    def __call__(self):
        print(self.counter)
        self.counter+=1
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
def compare_ensemble(in_path,
                     ens_factory,
                     n_splits=10,
                     n_repeats=1,
                     use_cpu=True):
    if(use_cpu):
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    data=dataset.read_csv(in_path)
    protocol=exp.AggrSplit(n_splits,n_repeats)
    splits=protocol.get_split(data)
    clfs=[("RF",exp.get_clf("RF")[1]),
          ("current_ens",ens_factory(data))]
    results={}
    for clf_type,clf_i in clfs:
        results[clf_type]=[split_k.eval(data,clf_i())  
                              for split_k in splits]
    for clf_type,results_i in results.items():
        print(results_i)
        acc=np.mean([result_j.get_acc() for result_j in results_i])
        balance=np.mean([result_j.get_balanced() for result_j in results_i])
        print(f"{clf_type},{acc},{balance}")

if __name__ == '__main__':
    compare_ensemble(in_path="../uci/wine-quality-red",
                     ens_factory=CEnsembleFactory)
#    deep_ens=ClassEnsemble()
#    nn,ens=compare_ensemble("../uci/cleveland",
#    	                    deep_ens=None,#deep_ens,
#    	                    single=False,
#    	                    verbose=0)
#    nn.report()
#    ens.report()
#    print(nn.get_acc())
#    print(ens.get_acc())
#    print(nn.get_balanced())
#    print(ens.get_balanced())
