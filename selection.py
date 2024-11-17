import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import pandas as pd
import dataset,utils

def feat_comp(in_path):
    result_dict=tree_impor(in_path,verbose=False)
    gini_dict={ data_i.split('/')[-1]: gini(arr_i) 
             for  data_i,arr_i in result_dict.items()}
    df=pd.DataFrame.from_records(data=list(gini_dict.items()),
    	                         columns=['data','gini'])
    df=df.sort_values(by='gini')
    print(df)

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

feat_comp("../uci")  
