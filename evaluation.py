import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score,balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
import utils

class OptimizedSVM(object):
    def __init__(self):
        rbf={'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]}
        linear={'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
        self.params=[rbf,linear]
        self.SVC=SVC(C=1)
        
    def gridSearch(self,X_train,y_train,metric='accuracy'):
        clf = gs.GridSearchCV(self.SVC,self.params, cv=5,scoring=metric)
        clf.fit(X_train,y_train)
        return clf

class OptimizedRandomForest(object):
    def __init__(self):
        params={}
        params['n_estimators']=[50,100,300,400,500] 
        #params['criterion']=['gini','entropy']
        self.params=[params]
        self.rf= RandomForestClassifier(n_estimators=10)
    
    def gridSearch(self,X_train,y_train,metric='accuracy'):
        clf = gs.GridSearchCV(self.rf,self.params, cv=5,scoring=metric)
        clf.fit(X_train,y_train)
        return clf

class OptimizedAdaBoost(object):
    def __init__(self):
        params={}
        params['n_estimators']=[50,100,150,200,300] 
        params['learning_rate']=[0.5,1.0,1.5,2.0]
        self.params=[params]
        self.ab=AdaBoostClassifier(n_estimators=100)

    def gridSearch(self,X_train,y_train,metric='accuracy'):
        clf = gs.GridSearchCV(self.ab,self.params, cv=5,scoring=metric)
        clf.fit(X_train,y_train)
        return clf

class Experiment(object):
    def __init__(self,features=None,clfs=None,score="acc",n_splits=10,n_repeats=1):
        if(clfs is None):
            clfs={"RF":RandomForestClassifier()}
        if(type(score)==str):
            score=get_score(score)
        self.features=features
        self.clfs=clfs
        self.score=score
        self.n_splits=n_splits
        self.n_repeats=n_repeats
    
    def __call__(self,in_path):
        if(self.features is None):
            dataset=utils.read_csv(in_path)
            data_dict={"base":dataset}
        else:
            data_dict=self.features(in_path)  
        result=self.eval(data_dict)
        result.score()

    def eval(self,data_dict):
        dataset=list(data_dict.values())[0]
    
        X,y=dataset.X,dataset.y
        rskf=RepeatedStratifiedKFold(n_repeats=self.n_repeats, 
                                     n_splits=self.n_splits, 
                                     random_state=0)
        results=Result()
        for name_i,data_i in data_dict.items():
            results[name_i]={name_j:[]  for name_j in self.clfs}
            for train_index,test_index in rskf.split(X,y):
                for name_j,clf_j in self.clfs.items():
                    y_pred,y_test=data_i.eval(train_index=train_index,
                                              test_index=test_index,
                                              clf=clf_j)
                    results.pairs_dict[name_i][name_j].append((y_pred,y_test))
        return results

class Result(object):
    def __init__(self):
        self.pairs_dict={}

    def __setitem__(self, key, value):
        self.pairs_dict[key]=value

    def score(self,score_type="acc"):
#        raise Exception(self.pairs_dict)
        score=get_score(score_type)
        for name_i,result_i in self.pairs_dict.items():
            for name_j,clf_j in result_i.items():
                metric_i=[score(true_j,pred_j) for true_j,pred_j in clf_j]
                metric_i=np.mean(metric_i)
                print(f"{name_i},{name_j}:{metric_i:.4f}")        

def pca_features(in_path):
    dataset=utils.read_csv(in_path)
    pca_data= utils.get_pca(dataset.X,dataset.y)
    return {"base":dataset,"pca":pca_data}

def evalOnTrainData(clf):
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))
    print()

def evalOnTestData(X_test,y_test,clf):
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))

def get_score(score_name:str):
    if(score_name=="balanced"):
        return balanced_accuracy_score
    return accuracy_score

exp=Experiment(features=pca_features)
exp("uci/cleveland")

#dataset=utils.read_csv("uci/cleveland")
#pca_data= utils.get_pca(dataset.X,dataset.y)
#data_dict={"base":dataset,"pca":dataset}
#random_eval(data_dict)
