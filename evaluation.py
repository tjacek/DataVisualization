# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 18:04:50 2015

@author: user
"""
#import sklearn.cross_validation as cv
#import sklearn.grid_search as gs
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report
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

def random_eval(dataset,n_split=5):
    X,y=dataset.X,dataset.y
    clf=RandomForestClassifier()
    scores = cross_val_score(clf, 
                             X, y, 
                             cv=n_split, 
                             scoring='accuracy')#'f1_macro')
    print(scores)


dataset=utils.read_csv("uci/cleveland")
random_eval(dataset)
pca_data= utils.get_pca(dataset.X,dataset.y)
random_eval(pca_data)