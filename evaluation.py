# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 18:04:50 2015

@author: user
"""
import arff
import sklearn.cross_validation as cv
import sklearn.grid_search as gs
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier

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

def evalSVM(dataset):
    X=dataset.data
    y=dataset.target
    X_train, X_test, y_train, y_test = cv.train_test_split(
                                       X, y, test_size=0.5, random_state=0)
    svm_opt=OptimizedAdaBoost()
    clf=svm_opt.gridSearch(X_train,y_train)
    
    evalOnTrainData(clf)
    evalOnTestData(X_test,y_test,clf)

prefix =  "C:/Users/user/Desktop/kwolek/DataVisualisation/data/"
name= prefix+"3_12_8.arff"   
dataset=arff.readArffDataset(name)
evalSVM(dataset)