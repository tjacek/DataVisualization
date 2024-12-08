import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
#from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import pandas as pd
import dataset,utils

def find_disc(in_path):
    df=pd.read_csv(in_path)
    y=df['target'].to_numpy()
    X=df[df.columns.difference(['data','target'])]
    X=X.to_numpy()
    clf=LogisticRegression()
    clf.fit(X,y)
    y_pred=clf.predict(X)
    print(y)
    print(y_pred)

def feat_comp(in_path):
    result_dict=tree_impor(in_path,verbose=False)
    gini_dict={ data_i.split('/')[-1]: gini(arr_i) 
             for  data_i,arr_i in result_dict.items()}
    
    for data_i,arr_i in result_dict.items():
        print(data_i)
#        print(arr_i)
        print(np.cumsum(np.flip(arr_i)))
    df=pd.DataFrame.from_records(data=list(gini_dict.items()),
    	                         columns=['data','gini'])
    df=df.sort_values(by='gini')
#    print(df)

@utils.DirFun({'in_path':0})
def tree_impor(in_path,verbose=True):
    data=dataset.read_csv(in_path)
    clf = ExtraTreesClassifier(n_estimators=50)
    clf = clf.fit(data.X, data.y)
    feat_imp=clf.feature_importances_
    feat_imp/=sum(feat_imp)
    feat_imp=np.sort(feat_imp)
    if(verbose):
        print(in_path)
        print(gini( feat_imp))
    return feat_imp

def gini(x):
    total = 0
    for i, xi in enumerate(x[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))
    return total / (len(x)**2 * np.mean(x))

find_disc(in_path="purity.csv")