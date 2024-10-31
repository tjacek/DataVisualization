import numpy as np

class Clustering(object):
    def __init__(self,dataset,cls_indices):
        if(type(cls_indices)==list):
            cls_indices=np.array(cls_indices)
        self.dataset=dataset
        self.cls_indices=cls_indices

    def n_clusters(self):
        return int(max(self.cls_indices))+1

    def get_cluster(self,i):
        clusters=[[] for _ in range(self.n_clusters())]
        for i,y_i in enumerate(self.dataset.y):
            cls_i=self.dataset.y[i]
            clusters[cls_i].append(y_i)
        return clusters

    def wihout_cluster(self,i):
        ind=(self.cls_indices==i)
        return self.dataset.selection(ind)

    def hist(self):
        hist=np.zeros((self.n_clusters(),
                       self.dataset.n_cats()))
        for i,clf_i in enumerate(self.cls_indices):
            y_i=int(self.dataset.y[i])
            hist[clf_i][y_i]+=1
        return Histogram(hist)
    
class Histogram(object):
    def __init__(self,arr):
        self.arr=arr
    
    def tp(self,cats=None):
        if(cats is None):
            cats=np.argmax(self.arr,axis=1)
        return np.array([self.arr[i][cat_i] 
                for i,cat_i in enumerate(cats)])

    def fp(self,tp=None):
        if(tp is None):
            tp=self.tp()
        cluster_sizes=np.sum(self.arr,axis=1)
        return cluster_sizes-tp

    def fn(self,cats=None,tp=None):
        if(cats is None):
            cats=np.argmax(self.arr,axis=1)
        if(tp is None):
            tp=self.tp(cats)
        cats_sizes=np.sum(self.arr,axis=0)
        return [ cats_sizes[cats[i]]-tp_i  
                 for i,tp_i in enumerate(tp)]
    
    def f1_score(self):
        cats=np.argmax(self.arr,axis=1)
        TP=self.tp(cats)
        FP,FN=self.fp(TP),self.fn(cats,TP)
        f1=[ (2.0*tp_i)/(2*tp_i+FP[i]+FN[i]) 
                   for i,tp_i in enumerate(TP)]
        return np.array(f1)
    
    def recall_matrix(self):
        n_clusters,n_cats=self.arr.shape  
        recall_matrix=[]
        for i in range(n_cats):
            cat_i=i*np.ones(n_clusters)
            cat_i=cat_i.astype(int)
            TP=self.tp(cat_i)
            FP,FN=self.fp(TP),self.fn(cat_i,TP)
            recall_i=[ tp_j/(tp_j+FN[j]) 
                       for j,tp_j in enumerate(TP)]
            recall_matrix.append(recall_i)
        return np.array(recall_matrix)

def ineq_measure(x):
    x=x/np.sum(x)
    return np.dot(x,x)