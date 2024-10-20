import deep,gauss,dataset

class Ensemble(object):
    def __init__(self):
        self.clfs=[]

    def predict(self,X):
        votes=[clf_i.predict(X)
                for clf_i in self.clfs]
        votes=np.array(votes)
        votes=np.sum(votes,axis=0)
        return np.argmax(votes,axis=1)

class GaussEnsemble(Ensemble):

	def fit(X,y):
		clustering=gauss.gaussian_clustering((X,y))
		n_cats=clustering.data.n_cats()
		for k in range(clustering.n_clusters()):
			cls_k=clustering.wihout_cluster(k)
			nn_k=deep,ClfCNN(default_cats=n_cats)
			nn_k.fit(X=cls_k.X,y=cls_k.y)
			self.clfs.append(nn_k)

def compare_ensemble(in_path,deep_ens=None):
    data=dataset.read_csv(in_path)
    if(deep_ens is None):
        deep_ens=GaussEnsemble()
    nn_k=deep,ClfCNN()