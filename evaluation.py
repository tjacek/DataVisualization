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

def evalSVM(dataset):
    X=dataset.data
    y=dataset.target
    X_train, X_test, y_train, y_test = cv.train_test_split(
                                       X, y, test_size=0.5, random_state=0)
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

   # score = 'precision'

    clf = gs.GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                       scoring='accuracy')
    clf.fit(X_train, y_train)
    
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))
    print()
    print("Best score: %0.3f" % clf.best_score_)
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))

prefix=    "C:/Users/user/Desktop/kwolek/DataVisualisation/data/"
name= prefix+"3_12_8.arff"   
dataset=arff.readArffDataset(name)
evalSVM(dataset)