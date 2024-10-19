import deep

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
		clustering=gaussian_clustering((X,y))