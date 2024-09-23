import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score,balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import dataset,utils

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
            data=dataset.read_csv(in_path)
            data_dict={"base":data}
        else:
            data_dict=self.features(in_path)  
        result=self.eval(data_dict)
        return result

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
        score=get_score(score_type)
        score_dict={}
        for name_i,result_i in self.pairs_dict.items():
            for name_j,clf_j in result_i.items():
                metric_i=[score(true_j,pred_j) for true_j,pred_j in clf_j]
                metric_i=np.mean(metric_i)
                print(f"{name_i},{name_j}:{metric_i:.4f}")
                score_dict[f"{name_i}_{name_j}"]=metric_i        
        return score_dict

    def report(self):
        classification_report(y_true, y_pred)  

    def to_df(self,metrics):
        if(type(metrics)==str):
            metrics=[metrics]
        all_dicts=[]
        for metric_i in metrics:
            all_dicts.append(self.score(metric_i))
        lines=[]
        for name_i in all_dicts[0].keys():
            lines.append(name_i.split("_"))
            for dict_j in all_dicts:
                lines[-1].append(dict_j[name_i])
        df=pd.DataFrame.from_records(lines)
        return df

def pca_features(in_path):
    data=dataset.read_csv(in_path)
    pca_data= dataset.get_pca(data.X,data.y)
    return {"base":data,"pca":pca_data}


def antr_features(in_path):
    data=dataset.read_csv(in_path)
    import autoencoder
    antr_data= autoencoder.AthroFeatures()(data)
    return {"base":data,"antr":antr_data}

def get_score(score_name:str):
    if(score_name=="balanced"):
        return balanced_accuracy_score
    return accuracy_score

def linear_exp():
    return Experiment(features=None,
                      clfs={"RF":RandomForestClassifier(),
                            "LR":LogisticRegression(solver='liblinear')})

if __name__ == '__main__':
    exp=linear_exp()#Experiment(features=antr_features)
    exp=utils.DirFun(exp)
    result=exp("uci")
    result.to_df(['acc'])