import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.metrics import accuracy_score,classification_report

class Dataset(object):
    def __init__(self,X,y=None):
        self.X=X
        self.y=y

    def __len__(self):
        return len(self.y)

    def dim(self):
        return self.X.shape[1]

    def n_cats(self):
        return int(max(self.y))+1

    def get_cat(self,i):
    	return self.X[self.y==i]
		
    def __call__(self,fun):
        return Dataset(X=fun(self.X),
                       y=self.y)
    
    def split(self,train_index,test_index):
        X_train=self.X[train_index]
        y_train=self.y[train_index]
        X_test=self.X[test_index]
        y_test=self.y[test_index]
        return (X_train,y_train),(X_test,y_test)        

    def eval(self,train_index,test_index,clf):
        (X_train,y_train),(X_test,y_test)=self.split(train_index,test_index)
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)
        return Result(y_pred,y_test)

    def labeled(self):
        return not (self.y is None)

    def min(self):
        return np.amin(self.X,axis=0)

    def max(self):
        return np.amax(self.X,axis=0)

    def class_weight(self):
        params={}
        for i in range(self.n_cats()):
            size_i= sum((self.y==i).astype(int))
            params[i]= 1.0/size_i
        Z= sum(list(params.values()))
        for i in params:
            params[i]= params[i]/Z
        return params

    def selection(self,indices):
        return Dataset(X=self.X[indices],
                       y=self.y[indices])

class Result(object):
    def __init__(self,y_pred,y_true):
        self.y_pred=y_pred
        self.y_true=y_true

    def get_acc(self):
        return accuracy_score(self.y_pred,self.y_true)

    def get_metric(self,metric):
        return metric(self.y_pred,self.y_true)

    def report(self):
        print(classification_report(self.y_pred,self.y_true,digits=4))

def read_result(in_path:str):
    raw=list(np.load(in_path).values())[0]
    y_pred,y_true=raw[0],raw[1]
    return Result(y_pred=y_pred,
                  y_true=y_true)

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
        return hist
    
    def stats(self):
        hist=self.hist()
        cats=np.argmax(hist,axis=1)
        cats_sizes=np.sum(hist,axis=0)
        cluster_sizes=np.sum(hist,axis=1)
        TP=np.array([hist[i][cat_i] 
                for i,cat_i in enumerate(cats)])
        FP= cluster_sizes-TP
        FN=[ cats_sizes[cats[i]]-tp_i  for i,tp_i in enumerate(TP)]
        return TP,FP,FN
    
    def f1_score(self):
        TP,FP,FN=self.stats()
        f1=[ (2.0*tp_i)/(2*tp_i+FP[i]+FN[i]) 
                   for i,tp_i in enumerate(TP)]
        return np.array(f1)

    def recall(self):
        TP,FP,FN=self.stats()
        recall=[ tp_i/(tp_i+FN[i]) 
                   for i,tp_i in enumerate(TP)]
        return np.array(recall)

def read_csv(in_path:str):
    if(type(in_path)==tuple):
        X,y=in_path
        return Dataset(X,y)
    df=pd.read_csv(in_path,header=None)
    raw=df.to_numpy()
    X,y=raw[:,:-1],raw[:,-1]
    X= preprocessing.RobustScaler().fit_transform(X)
    return Dataset(X,y)

def get_class_weights(y):
    params={}
    n_cats=int(max(y))+1
    for i in range(n_cats):
        size_i=(y==i).shape[0]
        params[i]= 1.0/size_i
    Z= sum(list(params.values()))
    for i in params:
        params[i]= params[i]/Z
    return params

def ineq_measure(x):
    x=x/np.sum(x)
    return np.dot(x,x)

if __name__ == '__main__':
    incomes = np.array([0,0,0,0,0,0,0,1000])#50, 50, 70, 70, 70, 90, 150, 150, 150, 150])
    print(gini_coefficient(incomes))
#    data=read_csv("../uci/lymphography")
#    for i in range(data.n_cats()):
#        x_i=data.get_cat(i)
#        print(x_i.shape)